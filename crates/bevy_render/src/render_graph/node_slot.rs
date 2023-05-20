use bevy_ecs::entity::Entity;
use std::{borrow::Cow, fmt};
use wgpu::{SamplerDescriptor, TextureFormat};

use crate::render_resource::{
    Buffer, BufferAddress, BufferUsages, Extent3d, Sampler, Texture, TextureDimension,
    TextureUsages, TextureView,
};
use crate::texture::CachedTexture;

/// A value passed between render [`Nodes`](super::Node).
/// Corresponds to the [`InputSlotDescriptor`] specified in the [`RenderGraph`](super::RenderGraph).
///
/// Slots can have four different types of values:
/// [`Buffer`], [`TextureView`], [`Sampler`] and [`Entity`].
///
/// These values do not contain the actual render data, but only the ids to retrieve them.
#[derive(Clone)]
pub enum SlotValue {
    /// A GPU-accessible [`Buffer`].
    Buffer(Buffer),
    /// A [`Texture`] describes a texture used in a pipeline.
    Texture(CachedTexture),
}

impl SlotValue {
    /// Returns the [`InputSlotDescriptor`] of this value.
    pub fn input_slot_descriptor(&self) -> InputSlotDescriptor {
        match self {
            SlotValue::Buffer(buffer) => InputSlotDescriptor::Buffer {
                buffer_usages: buffer.usage(),
            },
            SlotValue::Texture(_) => InputSlotDescriptor::Texture,
        }
    }
    /// Returns the [`InputSlotDescriptor`] of this value.
    pub fn computed_slot_descriptor(&self) -> ComputedSlotDescriptor {
        match self {
            SlotValue::Buffer(buffer) => ComputedSlotDescriptor::Buffer {
                size: buffer.size(),
                buffer_usages: buffer.usage(),
            },
            SlotValue::Texture(texture) => ComputedSlotDescriptor::Texture {
                size: texture.texture.size(),
                mip_level_count: texture.texture.mip_level_count(),
                sample_count: texture.texture.sample_count(),
                dimension: texture.texture.dimension(),
                format: texture.texture.format(),
                usage: texture.texture.usage(),
                view_formats: vec![texture.texture.format()], // TODO: How are view_formats determined?
            },
        }
    }
}

impl From<Buffer> for SlotValue {
    fn from(value: Buffer) -> Self {
        SlotValue::Buffer(value)
    }
}

/// A value passed between render [`Nodes`](super::Node).
/// Corresponds to the [`InputSlotDescriptor`] specified in the [`RenderGraph`](super::RenderGraph).
///
/// Slots can have four different types of values:
/// [`Buffer`], [`TextureView`], [`Sampler`] and [`Entity`].
///
/// These values do not contain the actual render data, but only the ids to retrieve them.
#[derive(Debug, Clone)]
pub enum OutputSlotValue {
    /// A GPU-accessible [`Buffer`].
    Buffer(Buffer),
    /// A [`Texture`] describes a texture used in a pipeline.
    Texture(Texture),
}

/// Describes the render resources created (output) or used (input) by
/// the render [`Nodes`](super::Node).
///
/// This should not be confused with [`SlotValue`], which actually contains the passed data.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InputSlotDescriptor {
    /// A GPU-accessible [`Buffer`].
    Buffer { buffer_usages: BufferUsages },
    /// A [`Texture`] describes a texture used in a pipeline.
    Texture,
}

/// Describes the render resources created (output) or used (input) by
/// the render [`Nodes`](super::Node).
///
/// This should not be confused with [`SlotValue`], which actually contains the passed data.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum ComputedSlotDescriptor {
    /// A GPU-accessible [`Buffer`].
    Buffer {
        size: BufferAddress,
        buffer_usages: BufferUsages,
    },
    /// A [`Texture`] describes a texture used in a pipeline.
    Texture {
        size: Extent3d,
        mip_level_count: u32,
        sample_count: u32,
        dimension: TextureDimension,
        format: TextureFormat,
        usage: TextureUsages,
        view_formats: Vec<TextureFormat>,
    },
}

impl fmt::Display for InputSlotDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            InputSlotDescriptor::Buffer { .. } => "Buffer",
            InputSlotDescriptor::Texture { .. } => "Texture",
        };

        f.write_str(s)
    }
}

/// A [`SlotLabel`] is used to reference a slot by either its name or index
/// inside the [`RenderGraph`](super::RenderGraph).
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SlotLabel {
    Index(usize),
    Name(Cow<'static, str>),
}

impl From<&SlotLabel> for SlotLabel {
    fn from(value: &SlotLabel) -> Self {
        value.clone()
    }
}

impl From<String> for SlotLabel {
    fn from(value: String) -> Self {
        SlotLabel::Name(value.into())
    }
}

impl From<&'static str> for SlotLabel {
    fn from(value: &'static str) -> Self {
        SlotLabel::Name(value.into())
    }
}

impl From<Cow<'static, str>> for SlotLabel {
    fn from(value: Cow<'static, str>) -> Self {
        SlotLabel::Name(value)
    }
}

impl From<usize> for SlotLabel {
    fn from(value: usize) -> Self {
        SlotLabel::Index(value)
    }
}

/// The internal representation of a slot, which specifies its [`InputSlotDescriptor`] and name.
#[derive(Clone, Debug)]
pub struct InputSlotInfo {
    pub name: Cow<'static, str>,
    pub slot_descriptor: InputSlotDescriptor,
}

impl InputSlotInfo {
    pub fn new(name: impl Into<Cow<'static, str>>, slot_descriptor: InputSlotDescriptor) -> Self {
        InputSlotInfo {
            name: name.into(),
            slot_descriptor,
        }
    }
}

/// The internal representation of a slot, which specifies its [`InputSlotDescriptor`] and name.
#[derive(Clone, Debug)]
pub struct OutputSlotInfo {
    pub name: Cow<'static, str>,
    pub slot_descriptor: ComputedSlotDescriptor,
}

impl OutputSlotInfo {
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        slot_descriptor: ComputedSlotDescriptor,
    ) -> Self {
        OutputSlotInfo {
            name: name.into(),
            slot_descriptor,
        }
    }
}

/// A collection of input or output [`SlotInfos`](SlotInfo) for
/// a [`NodeState`](super::NodeState).
#[derive(Default, Debug)]
pub struct InputSlotInfos {
    slots: Vec<InputSlotInfo>,
}

impl<T: IntoIterator<Item = InputSlotInfo>> From<T> for InputSlotInfos {
    fn from(slots: T) -> Self {
        InputSlotInfos {
            slots: slots.into_iter().collect(),
        }
    }
}

impl InputSlotInfos {
    /// Returns the count of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Returns true if there are no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Retrieves the [`SlotInfo`] for the provided label.
    pub fn get_slot(&self, label: impl Into<SlotLabel>) -> Option<&InputSlotInfo> {
        let label = label.into();
        let index = self.get_slot_index(label)?;
        self.slots.get(index)
    }

    /// Retrieves the [`SlotInfo`] for the provided label mutably.
    pub fn get_slot_mut(&mut self, label: impl Into<SlotLabel>) -> Option<&mut InputSlotInfo> {
        let label = label.into();
        let index = self.get_slot_index(label)?;
        self.slots.get_mut(index)
    }

    /// Retrieves the index (inside input or output slots) of the slot for the provided label.
    pub fn get_slot_index(&self, label: impl Into<SlotLabel>) -> Option<usize> {
        let label = label.into();
        match label {
            SlotLabel::Index(index) => Some(index),
            SlotLabel::Name(ref name) => self.slots.iter().position(|s| s.name == *name),
        }
    }

    /// Returns an iterator over the slot infos.
    pub fn iter(&self) -> impl Iterator<Item = &InputSlotInfo> {
        self.slots.iter()
    }
}
