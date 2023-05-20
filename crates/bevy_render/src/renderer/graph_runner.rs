use bevy_ecs::{prelude::Entity, world::World};
use bevy_utils::petgraph::algo::toposort;
use bevy_utils::petgraph::data::{Element, FromElements};
use bevy_utils::petgraph::visit::NodeRef;
use bevy_utils::petgraph::{Directed, Graph};
#[cfg(feature = "trace")]
use bevy_utils::tracing::info_span;
use bevy_utils::{HashMap, HashSet};
use smallvec::{smallvec, SmallVec};
#[cfg(feature = "trace")]
use std::ops::Deref;
use std::{borrow::Cow, collections::VecDeque};
use thiserror::Error;
use wgpu::TextureDescriptor;

use crate::render_graph::ComputedSlotDescriptor;
use crate::texture::{OwnedTextureDescriptor, TextureCache};
use crate::{
    render_graph::{
        Edge, InputSlotDescriptor, NodeId, NodeRunError, NodeState, RenderGraph,
        RenderGraphContext, SlotLabel, SlotValue,
    },
    renderer::{RenderContext, RenderDevice},
};

pub(crate) struct RenderGraphRunner;

#[derive(Error, Debug)]
pub enum RenderGraphRunnerError {
    #[error(transparent)]
    NodeRunError(#[from] NodeRunError),
    #[error("node output slot not set (index {slot_index}, name {slot_name})")]
    EmptyNodeOutputSlot {
        type_name: &'static str,
        slot_index: usize,
        slot_name: Cow<'static, str>,
    },
    #[error("graph (name: '{graph_name:?}') could not be run because slot '{slot_name}' at index {slot_index} has no value")]
    MissingInput {
        slot_index: usize,
        slot_name: Cow<'static, str>,
        graph_name: Option<Cow<'static, str>>,
    },
    #[error("attempted to use the wrong type for input slot")]
    MismatchedInputSlotType {
        slot_index: usize,
        label: SlotLabel,
        expected: InputSlotDescriptor,
        actual: InputSlotDescriptor,
    },
    #[error(
    "node (name: '{node_name:?}') has {slot_count} input slots, but was provided {value_count} values"
    )]
    MismatchedInputCount {
        node_name: Option<Cow<'static, str>>,
        slot_count: usize,
        value_count: usize,
    },
    #[error("graph has a node cycle")]
    CycleDetectedError,
}

#[derive(Copy, Clone)]
enum ValueSource {
    Input(usize),
    Node {
        local_node_id: usize,
        node_output_slot_id: usize,
    },
}

#[derive(Default, Clone)]
struct NodeExecutionState {
    input_sources: SmallVec<[Option<ValueSource>; 4]>,
    output_computed_source_descriptors: Option<Vec<ComputedSlotDescriptor>>,
    output_buffer_ids: Option<SmallVec<[usize; 4]>>,
    output_reference_counts: SmallVec<[usize; 4]>,
}

impl RenderGraphRunner {
    pub fn run(
        graph: &RenderGraph,
        render_device: RenderDevice,
        queue: &wgpu::Queue,
        world: &World,
        finalizer: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<(), RenderGraphRunnerError> {
        let mut render_context = RenderContext::new(render_device);
        Self::run_graph(graph, None, &mut render_context, world, &[], None)?;
        finalizer(render_context.command_encoder());

        {
            #[cfg(feature = "trace")]
            let _span = info_span!("submit_graph_commands").entered();
            queue.submit(render_context.finish());
        }
        Ok(())
    }

    fn run_graph(
        graph: &RenderGraph,
        graph_name: Option<Cow<'static, str>>,
        render_context: &mut RenderContext,
        world: &World,
        inputs: &[SlotValue],
        view_entity: Option<Entity>,
    ) -> Result<(), RenderGraphRunnerError> {
        let texture_cache = world.resource::<TextureCache>();
        #[cfg(feature = "trace")]
        let span = if let Some(name) = &graph_name {
            info_span!("run_graph", name = name.deref())
        } else {
            info_span!("run_graph", name = "main_graph")
        };
        #[cfg(feature = "trace")]
        let _guard = span.enter();

        // Reindex nodes
        let nodes_vec = graph.iter_nodes().collect::<Vec<_>>();
        let node_id_to_local_id = nodes_vec
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id, i))
            .collect::<bevy_utils::HashMap<NodeId, _>>();

        // Initialize execution states
        let mut node_execution_states: Vec<NodeExecutionState> =
            vec![Default::default(); nodes_vec.len()];
        for node_state in graph.iter_nodes() {
            let mut node_execution_state: &mut NodeExecutionState =
                &mut node_execution_states[node_id_to_local_id[&node_state.id]];
            node_execution_state.input_sources =
                SmallVec::from_elem(None, node_state.input_slots.len());
        }

        for edge in graph
            .iter_nodes()
            .flat_map(|node| node.edges.output_edges())
        {
            let Edge::SlotEdge {
                output_node,
                output_index,
                input_node,
                input_index,
            } = edge else { continue; };
            // Determine which node provides the value for each input slot
            node_execution_states[node_id_to_local_id[&input_node]].input_sources[*input_index] =
                Some(ValueSource::Node {
                    node_output_slot_id: *output_index,
                    local_node_id: node_id_to_local_id[output_node],
                });
            // Count the number of nodes referencing each output slot
            node_execution_states[node_id_to_local_id[&output_node]].output_reference_counts
                [*output_index] += 1;
        }

        // Set up the sources and output descriptors for the input node
        if let Some(input_node) = graph.get_input_node() {
            let mut input_slot_descriptors: SmallVec<[ComputedSlotDescriptor; 4]> = SmallVec::new();
            let mut sources = SmallVec::new();
            for (i, input_slot) in input_node.input_slots.iter().enumerate() {
                if let Some(input_value) = inputs.get(i) {
                    if input_slot.slot_descriptor != input_value.input_slot_descriptor() {
                        return Err(RenderGraphRunnerError::MismatchedInputSlotType {
                            slot_index: i,
                            actual: input_value.input_slot_descriptor(),
                            expected: input_slot.slot_descriptor,
                            label: input_slot.name.clone().into(),
                        });
                    }
                    input_slot_descriptors.push(input_value.computed_slot_descriptor());
                    sources.push(Some(ValueSource::Input(i)));
                } else {
                    return Err(RenderGraphRunnerError::MissingInput {
                        slot_index: i,
                        slot_name: input_slot.name.clone(),
                        graph_name,
                    });
                }
            }
            let local_id = node_id_to_local_id[&input_node.id];
            let mut node_execution_state = &mut node_execution_states[local_id];
            node_execution_state.input_sources = sources;
            node_execution_state.output_computed_source_descriptors = Some(
                input_node
                    .node
                    .get_outputs(input_slot_descriptors.as_slice()),
            );
        }
        // Create a PetGraph graph to perform a topological sorting
        let node_graph: Graph<&NodeState, (), Directed> =
            bevy_utils::petgraph::Graph::from_elements(
                nodes_vec
                    .iter()
                    .map(|node| Element::Node { weight: *node })
                    .chain(
                        nodes_vec
                            .iter()
                            .flat_map(|node| node.edges.output_edges())
                            .map(|edge| Element::Edge {
                                source: node_id_to_local_id[&edge.get_output_node()],
                                target: node_id_to_local_id[&edge.get_input_node()],
                                weight: (),
                            }),
                    ),
            );

        let Ok(sorted) = bevy_utils::petgraph::algo::toposort(&node_graph, None) else {
            // TODO: Better diagnostics
            return Err(RenderGraphRunnerError::CycleDetectedError);
        };

        // Calculate all slot descriptors
        for node_idx in &sorted {
            let node_state = node_graph.node_weight(*node_idx).unwrap();
            let output_local_id = node_id_to_local_id[&node_state.id];
            let mut execution_state: &NodeExecutionState = &node_execution_states[output_local_id];
            let input_shapes = execution_state.input_sources
                .iter()
                .enumerate()
                .map(|(i, source)| {
                    let source = source.ok_or(i)?;
                    Ok(match source {
                        ValueSource::Input(i) => inputs[i].computed_slot_descriptor(),
                        ValueSource::Node { local_node_id, node_output_slot_id } => {
                            node_execution_states[local_node_id]
                                .output_computed_source_descriptors
                                .as_ref()
                                .expect("output_computed_source_descriptors is None, incorrect topological ordering")
                                [node_output_slot_id]
                                .clone()
                        }
                    })
                })
                .collect::<Result<SmallVec<[_; 4]>, usize>>()
                .map_err(|i| RenderGraphRunnerError::MissingInput {
                    slot_index: i,
                    slot_name: node_state.input_slots.get_slot(i).unwrap().name.clone(),
                    graph_name: graph_name.clone(),
                })?;
            let mut execution_state: &mut NodeExecutionState =
                &mut node_execution_states[output_local_id];
            execution_state.output_computed_source_descriptors =
                Some(node_state.node.get_outputs(input_shapes.as_slice()));
        }
        // Map output slots to buffer/texture ids, recycling buffers/textures which are no longer
        // used.
        let mut buffer_id_counter = 0_usize;
        let mut textures = HashMap::new();
        let mut released_buffers: bevy_utils::HashMap<
            ComputedSlotDescriptor,
            SmallVec<[usize; 4]>,
        > = bevy_utils::HashMap::new();
        for node_idx in &sorted {
            let node_state = node_graph.node_weight(*node_idx).unwrap();
            let output_local_id = node_id_to_local_id[&node_state.id];
            let execution_state: &NodeExecutionState = &node_execution_states[output_local_id];
            let output_shapes = execution_state
                .output_computed_source_descriptors
                .as_ref()
                .unwrap();
            let mut output_buffers: SmallVec<[usize; 4]> = output_shapes
                .iter()
                .map(|output_shape|
                        // TODO: Create a common cache for textures and buffers
                        released_buffers.get_mut(output_shape)
                            .and_then(|matching_released_buffers| matching_released_buffers.pop())
                            .unwrap_or_else(|| {
                                let new_id = buffer_id_counter;
                                buffer_id_counter += 1;
                                match output_shape {
                                    ComputedSlotDescriptor::Buffer { .. } => {
                                        todo!()
                                    }
                                    ComputedSlotDescriptor::Texture {
                                        size,
                                        mip_level_count,
                                        sample_count,
                                        dimension,
                                        format,
                                        usage,
                                        view_formats
                                    } => {
                                        textures.insert(new_id, texture_cache.get(
                                            render_context.render_device(),
                                            OwnedTextureDescriptor {
                                                label: None,
                                                size: *size,
                                                mip_level_count: *mip_level_count,
                                                sample_count: *sample_count,
                                                dimension: *dimension,
                                                format: *format,
                                                usage: *usage,
                                                view_formats: view_formats.clone(),
                                            },
                                        ));
                                    }
                                }
                                new_id
                            }))
                .collect();
            let execution_state: &mut NodeExecutionState =
                &mut node_execution_states[output_local_id];
            execution_state.output_buffer_ids = Some(output_buffers);
            // After the node is executed, we can decrement the reference count for the node's
            // input slot buffer/texture. If its reference count reaches zero, we can reuse the
            // buffer/texture as outputs for the nodes to be executed later.
            for value_source in &node_execution_states[output_local_id].input_sources.clone() {
                let ValueSource::Node { local_node_id, node_output_slot_id } = value_source.unwrap() else { continue; };
                let output_execution_state: &mut NodeExecutionState =
                    &mut node_execution_states[local_node_id];
                output_execution_state.output_reference_counts[node_output_slot_id] -= 1;
                if output_execution_state.output_reference_counts[node_output_slot_id] == 0 {
                    released_buffers
                        .entry(
                            output_execution_state.output_computed_source_descriptors
                            .as_ref()
                            .unwrap() // This has been calculated at an earlier stage
                            [node_output_slot_id]
                                .clone(),
                        )
                        .or_insert_with(|| SmallVec::new())
                        .push(
                            output_execution_state.output_buffer_ids.as_ref().expect(
                                "output_buffer_ids is None, incorrect topological ordering",
                            )[node_output_slot_id],
                        );
                }
            }
        }
        // TODO: Prune unnecessary outputs

        // Execute the render graph
        for node_idx in &sorted {
            #[cfg(feature = "trace")]
            let _span = info_span!("node", name = node_state.type_name).entered();
            let node_state = node_graph.node_weight(*node_idx).unwrap();
            let output_local_id = node_id_to_local_id[&node_state.id];
            let execution_state: &NodeExecutionState = &node_execution_states[output_local_id];
            let mut outputs = execution_state
                .output_buffer_ids
                .as_ref()
                .unwrap()
                .iter()
                .map(|buffer| Some(SlotValue::Texture(textures[buffer].clone())))
                .collect::<Vec<Option<SlotValue>>>();
            let mut context = RenderGraphContext::new(graph, node_state, &inputs, &mut outputs);
            if let Some(view_entity) = view_entity {
                context.set_view_entity(view_entity);
            }
            node_state.node.run(&mut context, render_context, world)?;

            for run_sub_graph in context.finish() {
                let sub_graph = graph
                    .get_sub_graph(&run_sub_graph.name)
                    .expect("sub graph exists because it was validated when queued.");
                Self::run_graph(
                    sub_graph,
                    Some(run_sub_graph.name),
                    render_context,
                    world,
                    &run_sub_graph.inputs,
                    run_sub_graph.view_entity,
                )?;
            }
        }

        Ok(())
    }
}
