use std::collections::HashMap;

use quote::format_ident;

use crate::{types::TypesDefinitions, ModuleToTokensConfig};

fn make_global_binding(
    binding: &naga::ResourceBinding,
    _global: &naga::GlobalVariable,
    _module: &naga::Module,
) -> Vec<syn::Item> {
    let mut binding_items = Vec::new();

    let group = binding.group;
    let binding = binding.binding;
    binding_items.push(syn::Item::Const(syn::parse_quote! {
        pub const GROUP: u32 = #group;
    }));
    binding_items.push(syn::Item::Const(syn::parse_quote! {
        pub const BINDING: u32 = #binding;
    }));

    binding_items
}

fn address_space_to_tokens(address_space: naga::AddressSpace) -> proc_macro2::TokenStream {
    match address_space {
        naga::AddressSpace::Function => quote::quote!(naga::AddressSpace::Function),
        naga::AddressSpace::Private => quote::quote!(naga::AddressSpace::Private),
        naga::AddressSpace::WorkGroup => quote::quote!(naga::AddressSpace::WorkGroup),
        naga::AddressSpace::Uniform => quote::quote!(naga::AddressSpace::Uniform),
        naga::AddressSpace::Storage { access } => {
            let bits = access.bits();
            quote::quote!(naga::AddressSpace::Storage {
                access: naga::StorageAccess::from_bits_retain(#bits)
            })
        }
        naga::AddressSpace::Handle => quote::quote!(naga::AddressSpace::Handle),
        naga::AddressSpace::PushConstant => quote::quote!(naga::AddressSpace::PushConstant),
    }
}

fn image_dimension_to_tokens(
    image_dimension: naga::ImageDimension,
    arrayed: bool,
) -> proc_macro2::TokenStream {
    match image_dimension {
        naga::ImageDimension::D1 => quote::quote! { ::wgpu::TextureViewDimension::D1 },
        naga::ImageDimension::D2 => {
            if arrayed {
                quote::quote! { ::wgpu::TextureViewDimension::D2Array }
            } else {
                quote::quote! { ::wgpu::TextureViewDimension::D2 }
            }
        }
        naga::ImageDimension::D3 => quote::quote! { ::wgpu::TextureViewDimension::D3 },
        naga::ImageDimension::Cube => {
            if arrayed {
                quote::quote! { ::wgpu::TextureViewDimension::CubeArray }
            } else {
                quote::quote! { ::wgpu::TextureViewDimension::Cube }
            }
        }
    }
}

fn image_class_to_sample_type_tokens(class: naga::ImageClass) -> proc_macro2::TokenStream {
    match class {
        naga::ImageClass::Depth { .. } => quote::quote! { ::wgpu::TextureSampleType::Depth },
        naga::ImageClass::Sampled { kind, .. } => match kind {
            naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat => {
                quote::quote! { ::wgpu::TextureSampleType::Float { filterable: true } }
            } // TODO: support non-filtering?
            naga::ScalarKind::Sint => quote::quote! { ::wgpu::TextureSampleType::Sint },
            naga::ScalarKind::Uint => quote::quote! { ::wgpu::TextureSampleType::Uint },
            _ => todo!(),
        },
        _ => todo!(),
    }
}

/// Converts a global in a module into a collection of Rust definitions including the type and binding of the global,
/// if representable.
pub fn make_global(
    global: &naga::GlobalVariable,
    module: &naga::Module,
    types: &mut TypesDefinitions,
    args: &ModuleToTokensConfig,
) -> Vec<syn::Item> {
    let mut global_items = Vec::new();

    if let Some(name) = &global.name {
        global_items.push(syn::Item::Const(syn::parse_quote! {
            pub const NAME: &'static str = #name;
        }));
    }

    if args.gen_naga {
        let space = address_space_to_tokens(global.space);
        global_items.push(syn::Item::Const(syn::parse_quote! {
            #[allow(unused)]
            pub const SPACE: naga::AddressSpace = #space;
        }));
    }

    if let Some(type_ident) = types.rust_type_ident(global.ty, module, args) {
        global_items.push(syn::Item::Type(syn::parse_quote! {
            pub type Ty = #type_ident;
        }));
    }

    if let Some(binding) = &global.binding {
        let binding_items = make_global_binding(binding, global, module);
        global_items.extend(binding_items.into_iter());
    }

    global_items
}

/// Generate a function that builds a wgpu::BindGroupLayoutDescriptor for the associated group of
/// bindings.
pub fn make_group_layout_fn(
    module: &naga::Module,
    group: u32,
    group_name: &str,
    resource_bindings: &[&naga::GlobalVariable],
) -> syn::Item {
    let entries = resource_bindings.iter().enumerate().map(|(binding, v)| {
        let binding = binding as u32;
        let ty = &module.types[v.ty];
         match ty.inner {
            naga::TypeInner::Scalar(_) => todo!(),
            naga::TypeInner::Vector { .. } => todo!(),
            naga::TypeInner::Matrix {
                ..
            } => todo!(),
            naga::TypeInner::Atomic(_) => todo!(),
            naga::TypeInner::Pointer { .. } => todo!(),
            naga::TypeInner::ValuePointer {
                ..
            } => todo!(),
            naga::TypeInner::Array { .. } => todo!(),
            naga::TypeInner::Struct { .. } => {
                assert!(v.space == naga::AddressSpace::Uniform);
                quote::quote! {
                    ::wgpu::BindGroupLayoutEntry {
                        binding: #binding,
                        visibility: ::wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: ::wgpu::BindingType::Buffer {
                            ty: ::wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                }
            }
            naga::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let view_dimension = image_dimension_to_tokens(dim, arrayed);
                let multisampled = class.is_multisampled();
                let sample_type = image_class_to_sample_type_tokens(class);
                quote::quote! {
                    ::wgpu::BindGroupLayoutEntry {
                        binding: #binding,
                        visibility: ::wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: #multisampled,
                            view_dimension: #view_dimension,
                            sample_type: #sample_type,
                        },
                        count: None,
                    }
                }
            }
            naga::TypeInner::Sampler { comparison } => {
                let sampler_type = if comparison {
                    quote::quote!(::wgpu::SamplerBindingType::Comparison)
                } else {
                    let sample_type_is_float = resource_bindings.iter().find_map(|b| {
                        let ty = &module.types[b.ty];
                        match ty.inner {
                            naga::TypeInner::Image { class: naga::ImageClass::Depth { .. }, .. } => Some(true),
                            naga::TypeInner::Image { class: naga::ImageClass::Sampled { kind, .. }, .. } => Some(match kind {
                                naga::ScalarKind::Float | naga::ScalarKind::AbstractFloat => true,
                                _ => false,
                            }),
                            _ => None,
                        }
                    }).expect(&format!(
                        "could not find matching texture for sampler {:?} - @group({}) @binding({})",
                        v.name,
                        group,
                        binding,
                    ));
                    if sample_type_is_float {
                        quote::quote!(::wgpu::SamplerBindingType::Filtering)
                    } else {
                        quote::quote!(::wgpu::SamplerBindingType::NonFiltering)
                    }
                };
                quote::quote! {
                    ::wgpu::BindGroupLayoutEntry {
                        binding: #binding,
                        visibility: ::wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: ::wgpu::BindingType::Sampler(#sampler_type),
                        count: None,
                    }
                }
            }
            naga::TypeInner::AccelerationStructure => todo!(),
            naga::TypeInner::RayQuery => todo!(),
            naga::TypeInner::BindingArray { .. } => todo!(),
        }
    });
    syn::parse_quote! {
        pub fn layout() -> ::wgpu::BindGroupLayoutDescriptor<'static> {
            ::wgpu::BindGroupLayoutDescriptor {
                label: Some(#group_name),
                entries: &[
                    #(#entries),*
                ],
            }
        }
    }
}

/// Builds a collection of globals into a collection of Rust module definitions containing
/// each of the globals' properties, such as type and binding.
pub fn make_globals(
    module: &naga::Module,
    types: &mut TypesDefinitions,
    args: &ModuleToTokensConfig,
) -> Vec<syn::Item> {
    let mut globals = Vec::new();

    // Info about all globals together
    let mut groups = HashMap::new();
    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding {
            groups.entry(binding.group).or_insert(vec![]).push(global)
        }
    }

    // Info about each global individually
    for (_, global) in module.global_variables.iter() {
        // Get name for global module
        let Some(global_name) = &global.name else {
            continue;
        };
        let Ok(global_name_ident) = syn::parse_str::<syn::Ident>(global_name) else {
            continue;
        };

        // Make items within module
        let global_items = crate::collect_tokenstream(make_global(global, module, types, args));

        // Collate into an inner module
        let doc = format!(
            "Information about the `{}` global variable within this shader module.",
            global_name
        );
        let group = global.binding.as_ref().unwrap().group;
        let layout_fn = (groups[&group].len() == 1).then(|| {
            let bindings = groups.remove(&group).unwrap();
            // this is the only variable in the group
            make_group_layout_fn(module, group, &global_name, &bindings)
        });
        globals.push(syn::parse_quote! {
            #[doc = #doc]
            pub mod #global_name_ident {
                #[allow(unused)]
                use super::*;

                #global_items
                #layout_fn
            }
        })
    }
    for (group, mut bindings) in groups.drain() {
        assert!(bindings.len() > 1);
        let group_ident = format_ident!("group{group}");
        bindings.sort_by_key(|x| x.binding.as_ref().unwrap().binding);
        let layout_fn = make_group_layout_fn(module, group, &group_ident.to_string(), &bindings);
        let doc = format!(
            "Contains the following bindings: {}",
            bindings
                .into_iter()
                .map(|b| b.name.clone().expect("binding has no name"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        globals.push(syn::parse_quote! {
            #[doc = #doc]
            pub mod #group_ident {
                #[allow(unused)]
                use super::*;

                #layout_fn
            }
        });
    }
    //TODO: Create `create_bind_groups` ctr function

    globals
}
