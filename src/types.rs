use std::collections::{HashMap, HashSet};

use naga::proc::{Alignment, Layouter};
use proc_macro2::TokenStream;
use quote::format_ident;

use crate::ModuleToTokensConfig;

/// Returns a base Rust or `glam` type that corresponds to a TypeInner, if one exists.
fn rust_type(type_inner: &naga::TypeInner, args: &ModuleToTokensConfig) -> Option<syn::Type> {
    match type_inner {
        naga::TypeInner::Scalar(naga::Scalar { kind, width }) => match (kind, width) {
            (naga::ScalarKind::Bool, 1) => Some(syn::parse_quote!(bool)),
            (naga::ScalarKind::Float, 4) => Some(syn::parse_quote!(f32)),
            (naga::ScalarKind::Float, 8) => Some(syn::parse_quote!(f64)),
            (naga::ScalarKind::Sint, 4) => Some(syn::parse_quote!(i32)),
            (naga::ScalarKind::Sint, 8) => Some(syn::parse_quote!(i64)),
            (naga::ScalarKind::Uint, 4) => Some(syn::parse_quote!(u32)),
            (naga::ScalarKind::Uint, 8) => Some(syn::parse_quote!(u64)),
            _ => None,
        },
        naga::TypeInner::Vector {
            size,
            scalar: naga::Scalar { kind, width },
        } => {
            if args.gen_glam {
                match (size, kind, width) {
                    (naga::VectorSize::Bi, naga::ScalarKind::Bool, 1) => {
                        Some(syn::parse_quote!(glam::bool::BVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Bool, 1) => {
                        Some(syn::parse_quote!(glam::bool::BVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Bool, 1) => {
                        Some(syn::parse_quote!(glam::bool::BVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Float, 4) => {
                        Some(syn::parse_quote!(glam::f32::Vec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Float, 4) => {
                        Some(syn::parse_quote!(glam::f32::Vec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Float, 4) => {
                        Some(syn::parse_quote!(glam::f32::Vec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Float, 8) => {
                        Some(syn::parse_quote!(glam::f64::DVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Float, 8) => {
                        Some(syn::parse_quote!(glam::f64::DVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Float, 8) => {
                        Some(syn::parse_quote!(glam::f64::DVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Sint, 4) => {
                        Some(syn::parse_quote!(glam::i32::IVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Sint, 4) => {
                        Some(syn::parse_quote!(glam::i32::IVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Sint, 4) => {
                        Some(syn::parse_quote!(glam::i32::IVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Sint, 8) => {
                        Some(syn::parse_quote!(glam::i64::I64Vec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Sint, 8) => {
                        Some(syn::parse_quote!(glam::i64::I64Vec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Sint, 8) => {
                        Some(syn::parse_quote!(glam::i64::I64Vec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Uint, 4) => {
                        Some(syn::parse_quote!(glam::u32::UVec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Uint, 4) => {
                        Some(syn::parse_quote!(glam::u32::UVec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Uint, 4) => {
                        Some(syn::parse_quote!(glam::u32::UVec4))
                    }
                    (naga::VectorSize::Bi, naga::ScalarKind::Uint, 8) => {
                        Some(syn::parse_quote!(glam::u64::U64Vec2))
                    }
                    (naga::VectorSize::Tri, naga::ScalarKind::Uint, 8) => {
                        Some(syn::parse_quote!(glam::u64::U64Vec3))
                    }
                    (naga::VectorSize::Quad, naga::ScalarKind::Uint, 8) => {
                        Some(syn::parse_quote!(glam::u64::U64Vec4))
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar: naga::Scalar { kind, width },
        } => {
            if !args.gen_glam {
                return None;
            }
            if columns != rows {
                return None;
            }
            match (kind, width) {
                (naga::ScalarKind::Float, 4) => match columns {
                    naga::VectorSize::Bi => Some(syn::parse_quote!(glam::f32::Mat2)),
                    naga::VectorSize::Tri => Some(syn::parse_quote!(glam::f32::Mat3)),
                    naga::VectorSize::Quad => Some(syn::parse_quote!(glam::f32::Mat4)),
                },
                (naga::ScalarKind::Float, 8) => match columns {
                    naga::VectorSize::Bi => Some(syn::parse_quote!(glam::f64::Mat2)),
                    naga::VectorSize::Tri => Some(syn::parse_quote!(glam::f64::Mat3)),
                    naga::VectorSize::Quad => Some(syn::parse_quote!(glam::f64::Mat4)),
                },
                _ => None,
            }
        }
        naga::TypeInner::Atomic(scalar) => rust_type(&naga::TypeInner::Scalar(*scalar), args),
        _ => None,
    }
}

fn vertex_format(type_inner: &naga::TypeInner) -> syn::Field {
    match type_inner {
        naga::TypeInner::Scalar(naga::Scalar { kind, width }) => match (kind, width) {
            (naga::ScalarKind::Bool, 1) => todo!(),
            (naga::ScalarKind::Float, 4) => syn::parse_quote!(::wgpu::VertexFormat::Float32),
            (naga::ScalarKind::Float, 8) => syn::parse_quote!(::wgpu::VertexFormat::Float64),
            (naga::ScalarKind::Sint, 4) => syn::parse_quote!(::wgpu::VertexFormat::Sint32),
            (naga::ScalarKind::Sint, 8) => todo!(),
            (naga::ScalarKind::Uint, 4) => syn::parse_quote!(::wgpu::VertexFormat::Uint32),
            (naga::ScalarKind::Uint, 8) => todo!(),
            _ => todo!(),
        },
        naga::TypeInner::Vector {
            size,
            scalar: naga::Scalar { kind, width },
        } => match (size, kind, width) {
            // Float vectors
            (naga::VectorSize::Bi, naga::ScalarKind::Float, 2) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float16x2)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Float, 2) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float16x4)
            }
            (naga::VectorSize::Bi, naga::ScalarKind::Float, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float32x2)
            }
            (naga::VectorSize::Tri, naga::ScalarKind::Float, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float32x3)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Float, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float32x4)
            }
            (naga::VectorSize::Bi, naga::ScalarKind::Float, 8) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float64x2)
            }
            (naga::VectorSize::Tri, naga::ScalarKind::Float, 8) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float64x3)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Float, 8) => {
                syn::parse_quote!(::wgpu::VertexFormat::Float64x4)
            }
            // Signed int vectors
            (naga::VectorSize::Bi, naga::ScalarKind::Sint, 1) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint8x2)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Sint, 1) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint8x4)
            }
            (naga::VectorSize::Bi, naga::ScalarKind::Sint, 2) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint16x2)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Sint, 2) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint16x4)
            }
            (naga::VectorSize::Bi, naga::ScalarKind::Sint, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint32x2)
            }
            (naga::VectorSize::Tri, naga::ScalarKind::Sint, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint32x3)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Sint, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Sint32x4)
            }
            // Unsigned int vectors
            (naga::VectorSize::Bi, naga::ScalarKind::Uint, 1) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint8x4)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Uint, 1) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint8x4)
            }
            (naga::VectorSize::Bi, naga::ScalarKind::Uint, 2) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint16x4)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Uint, 2) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint16x4)
            }
            (naga::VectorSize::Bi, naga::ScalarKind::Uint, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint32x4)
            }
            (naga::VectorSize::Tri, naga::ScalarKind::Uint, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint32x4)
            }
            (naga::VectorSize::Quad, naga::ScalarKind::Uint, 4) => {
                syn::parse_quote!(::wgpu::VertexFormat::Uint32x4)
            }
            // TODO: normalized types - maybe with an attribute tag?
            _ => todo!(),
        },
        naga::TypeInner::Matrix { .. } => {
            // error
            todo!();
        }
        // naga::TypeInner::Atomic(scalar) => rust_type(&naga::TypeInner::Scalar(*scalar), args),
        _ => todo!(),
    }
}

/// A builder for type definition and identifier pairs.
pub struct TypesDefinitions {
    items: Vec<syn::Item>,
    references: HashMap<naga::Handle<naga::Type>, syn::Type>,
    structs_filter: Option<HashSet<String>>,
    vertex_input_types: Option<HashSet<String>>,
}

impl TypesDefinitions {
    /// Constructs a new type definition collator, with a given filter for type names.
    pub fn new(module: &naga::Module, args: &ModuleToTokensConfig) -> Self {
        let mut res = Self {
            items: Vec::new(),
            references: HashMap::new(),
            structs_filter: args.structs_filter.clone(),
            vertex_input_types: args.vertex_input_types.clone(),
        };

        for (ty_handle, _) in module.types.iter() {
            if let Some(new_ty_ident) = res.try_make_type(ty_handle, module, args) {
                res.references.insert(ty_handle, new_ty_ident.clone());
            }
        }

        res
    }

    fn try_make_type(
        &mut self,
        ty_handle: naga::Handle<naga::Type>,
        module: &naga::Module,
        args: &ModuleToTokensConfig,
    ) -> Option<syn::Type> {
        let ty = module.types.get_handle(ty_handle).ok()?;
        if let Some(ty_ident) = rust_type(&ty.inner, args) {
            return Some(ty_ident);
        };

        if let Some(repl) = ty
            .name
            .as_ref()
            .and_then(|name| args.type_overrides.get(name))
        {
            return Some(repl.clone());
        }

        match &ty.inner {
            naga::TypeInner::Array { base, size, .. }
            | naga::TypeInner::BindingArray { base, size } => {
                let base_type = self.rust_type_ident(*base, module, args)?;
                match size {
                    naga::ArraySize::Constant(size) => {
                        let size = size.get();
                        Some(syn::parse_quote!([#base_type; #size as usize]))
                    }
                    naga::ArraySize::Dynamic => Some(syn::parse_quote!(Vec<#base_type>)),
                    naga::ArraySize::Pending(_) => todo!(),
                }
            }
            naga::TypeInner::Struct { members, .. } => {
                let struct_name = ty.name.as_ref();
                let struct_name = match struct_name {
                    None => return None,
                    Some(struct_name) => struct_name,
                };

                // Apply filter
                if let Some(struct_name_filter) = &self.structs_filter {
                    if !struct_name_filter.contains(struct_name) {
                        return None;
                    }
                }

                let is_vertex_input = self
                    .vertex_input_types
                    .as_ref()
                    .is_some_and(|vertex_input_types| vertex_input_types.contains(struct_name));

                let mut layouter = Layouter::default();
                layouter.update(module.to_ctx()).unwrap();

                let members_have_names = members.iter().all(|member| member.name.is_some());
                let mut last_field_name = None;
                let mut total_offset = 0;
                let mut largest_alignment = 0;
                let mut member_types = vec![];

                let mut members: Vec<_> = members
                    .iter()
                    .enumerate()
                    .map(|(i_member, member)| {
                        let member_name = if members_have_names {
                            let member_name =
                                member.name.as_ref().expect("all members had names").clone();
                            syn::parse_str::<syn::Ident>(&member_name)
                        } else {
                            syn::parse_str::<syn::Ident>(&format!("v{}", i_member))
                        };
                        let member_ty = self.rust_type_ident(member.ty, module, args);

                        let mut attributes = proc_macro2::TokenStream::new();
                        // Runtime-sized fields must be marked as such when using encase
                        if args.gen_encase {
                            let ty = module.types.get_handle(member.ty);
                            if let Ok(naga::Type {
                                inner:
                                    naga::TypeInner::Array {
                                        size: naga::ArraySize::Dynamic,
                                        ..
                                    }
                                    | naga::TypeInner::BindingArray {
                                        size: naga::ArraySize::Dynamic,
                                        ..
                                    },
                                ..
                            }) = ty
                            {
                                attributes.extend(quote::quote!(#[size(runtime)]))
                            }
                        }

                        member_ty.and_then(|member_ty| {
                            member_name.ok().map(|member_name| {
                                let inner_type = module
                                    .types
                                    .get_handle(member.ty)
                                    .expect("failed to locate member type")
                                    .inner
                                    .clone();
                                let field_size = inner_type.size(module.to_ctx());
                                member_types.push((inner_type, field_size));
                                let alignment = layouter[member.ty].alignment;
                                largest_alignment = largest_alignment.max(alignment * 1u32);
                                let padding_needed =
                                    if is_vertex_input || alignment.is_aligned(total_offset) {
                                        0
                                    } else {
                                        alignment.round_up(total_offset) - total_offset
                                    };
                                let pad = if padding_needed > 0 {
                                    let padding_member_name = format_ident!(
                                        "_pad_{}",
                                        last_field_name.as_ref().expect(
                                            "invariant: expected prior member before padding field"
                                        )
                                    );
                                    quote::quote! {
                                        pub #padding_member_name: [u8; #padding_needed as usize],
                                    }
                                } else {
                                    quote::quote! {}
                                };
                                total_offset += field_size + padding_needed;
                                last_field_name = Some(member_name.clone());
                                quote::quote! {
                                    #pad
                                    #attributes
                                    pub #member_name: #member_ty
                                }
                            })
                        })
                    })
                    .collect();
                let struct_alignment = Alignment::from_width(largest_alignment as u8);
                if !is_vertex_input && !struct_alignment.is_aligned(total_offset) {
                    // struct needs padding to be aligned
                    let padding_needed = struct_alignment.round_up(total_offset) - total_offset;
                    members.push(Some(quote::quote! {
                        pub _pad: [u8; #padding_needed as usize],
                    }));
                }
                let struct_name = syn::parse_str::<syn::Ident>(struct_name).ok();
                match (members, struct_name) {
                    (members, Some(struct_name)) => {
                        #[allow(unused_mut)]
                        let mut bonus_struct_derives = TokenStream::new();
                        if args.gen_encase {
                            bonus_struct_derives.extend(quote::quote!(encase::ShaderType,))
                        }
                        if args.gen_bytemuck {
                            bonus_struct_derives
                                .extend(quote::quote!(bytemuck::Pod, bytemuck::Zeroable,));
                        }

                        self.items.push(syn::parse_quote! {
                            #[allow(unused, non_camel_case_types)]
                            #[repr(C)]
                            #[derive(Debug, PartialEq, Copy, Clone, #bonus_struct_derives)]
                            pub struct #struct_name {
                                #(#members),*
                            }
                        });
                        if is_vertex_input {
                            let n_attributes = members.len();
                            let mut offset: u64 = 0;
                            let attributes =
                                member_types.iter().zip(0u32..).map(|((ty, size), i)| {
                                    let format = vertex_format(ty);
                                    let x = quote::quote! {
                                        ::wgpu::VertexAttribute{
                                            format: #format,
                                            offset: #offset,
                                            shader_location: #i,
                                        }
                                    };
                                    offset += *size as u64;
                                    x
                                });
                            self.items.push(syn::parse_quote! {
                                // This is a vertex input type
                                impl #struct_name {
                                    pub const VERTEX_ATTRIBUTES: [::wgpu::VertexAttribute; #n_attributes] = [
                                        #(#attributes),*
                                    ];

                                    pub fn vertex_buffer_layout() -> ::wgpu::VertexBufferLayout<'static> {
                                        ::wgpu::VertexBufferLayout {
                                            array_stride: ::std::mem::size_of::<Self>() as ::wgpu::BufferAddress,
                                            step_mode: ::wgpu::VertexStepMode::Vertex,
                                            attributes: &Self::VERTEX_ATTRIBUTES,
                                        }
                                    }
                                }
                            });
                        }
                        Some(syn::parse_quote!(#struct_name))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Takes a handle to a type, and a module where the type resides, and tries to return an identifier
    /// of that type, in Rust. Note that for structs this will be an identifier in to the set of structs generated
    /// by calling `TypesDefinitions::definitions()`, so your output should make sure to include everything from
    /// there in the scope where the returned identifier is used.
    pub fn rust_type_ident(
        &mut self,
        ty_handle: naga::Handle<naga::Type>,
        module: &naga::Module,
        args: &ModuleToTokensConfig,
    ) -> Option<syn::Type> {
        if let Some(ident) = self.references.get(&ty_handle).cloned() {
            return Some(ident);
        }

        if let Some(built) = self.try_make_type(ty_handle, module, args) {
            self.references.insert(ty_handle, built.clone());
            return Some(built);
        }

        None
    }

    /// Gives the set of definitions required by the identifiers generated by this object. These should be
    /// emitted somewhere accessable by the places that the identifiers were used.
    pub fn items(self) -> Vec<syn::Item> {
        self.items
    }
}
