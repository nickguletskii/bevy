use crate::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use crate::{
    render_resource::{Texture, TextureView},
    renderer::RenderDevice,
};
use bevy_ecs::{prelude::ResMut, system::Resource};
use bevy_utils::{Entry, HashMap};
use std::pin::Pin;
use wgpu::{TextureDescriptor, TextureViewDescriptor};

/// The internal representation of a [`CachedTexture`] used to track whether it was recently used
/// and is currently taken.
struct CachedTextureMeta {
    texture: Texture,
    default_view: TextureView,
    taken: bool,
    frames_since_last_use: usize,
}

/// A cached GPU [`Texture`] with corresponding [`TextureView`].
/// This is useful for textures that are created repeatedly (each frame) in the rendering process
/// to reduce the amount of GPU memory allocations.
#[derive(Clone)]
pub struct CachedTexture {
    pub texture: Texture,
    pub default_view: TextureView,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct OwnedTextureDescriptor {
    pub label: Option<String>,
    pub size: Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsages,
    pub view_formats: Vec<TextureFormat>,
}

/// This resource caches textures that are created repeatedly in the rendering process and
/// are only required for one frame.
#[derive(Resource, Default)]
pub struct TextureCache {
    textures: std::sync::Mutex<HashMap<OwnedTextureDescriptor, Vec<CachedTextureMeta>>>,
}

impl TextureCache {
    /// Retrieves a texture that matches the `descriptor`. If no matching one is found a new
    /// [`CachedTexture`] is created.
    pub fn get(
        &self,
        render_device: &RenderDevice,
        descriptor: OwnedTextureDescriptor,
    ) -> CachedTexture {
        let mut guard = self.textures.lock().unwrap();
        match guard.entry(descriptor.clone()) {
            Entry::Occupied(mut entry) => {
                for texture in entry.get_mut().iter_mut() {
                    if !texture.taken {
                        texture.frames_since_last_use = 0;
                        texture.taken = true;
                        return CachedTexture {
                            texture: texture.texture.clone(),
                            default_view: texture.default_view.clone(),
                        };
                    }
                }

                let texture = render_device.create_texture(&TextureDescriptor {
                    label: None,
                    size: descriptor.size,
                    mip_level_count: descriptor.mip_level_count,
                    sample_count: descriptor.sample_count,
                    dimension: descriptor.dimension,
                    format: descriptor.format,
                    usage: descriptor.usage,
                    view_formats: descriptor.view_formats.as_slice(),
                });
                let default_view = texture.create_view(&TextureViewDescriptor::default());
                entry.get_mut().push(CachedTextureMeta {
                    texture: texture.clone(),
                    default_view: default_view.clone(),
                    frames_since_last_use: 0,
                    taken: true,
                });
                CachedTexture {
                    texture,
                    default_view,
                }
            }
            Entry::Vacant(entry) => {
                let descriptor = entry.key();
                let texture = render_device.create_texture(&TextureDescriptor {
                    label: None,
                    size: descriptor.size,
                    mip_level_count: descriptor.mip_level_count,
                    sample_count: descriptor.sample_count,
                    dimension: descriptor.dimension,
                    format: descriptor.format,
                    usage: descriptor.usage,
                    view_formats: descriptor.view_formats.as_slice(),
                });
                let default_view = texture.create_view(&TextureViewDescriptor::default());
                entry.insert(vec![CachedTextureMeta {
                    texture: texture.clone(),
                    default_view: default_view.clone(),
                    taken: true,
                    frames_since_last_use: 0,
                }]);
                CachedTexture {
                    texture,
                    default_view,
                }
            }
        }
    }

    /// Updates the cache and only retains recently used textures.
    pub fn update(&mut self) {
        let mut guard = self.textures.lock().unwrap();
        for textures in guard.values_mut() {
            for texture in textures.iter_mut() {
                texture.frames_since_last_use += 1;
                texture.taken = false;
            }

            textures.retain(|texture| texture.frames_since_last_use < 3);
        }
    }
}

/// Updates the [`TextureCache`] to only retains recently used textures.
pub fn update_texture_cache_system(mut texture_cache: ResMut<TextureCache>) {
    texture_cache.update();
}
