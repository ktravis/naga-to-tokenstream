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

/// A builder for type definition and identifier pairs.
pub struct TypesDefinitions {
    items: Vec<syn::Item>,
    references: HashMap<naga::Handle<naga::Type>, syn::Type>,
    structs_filter: Option<HashSet<String>>,
}

impl TypesDefinitions {
    /// Constructs a new type definition collator, with a given filter for type names.
    pub fn new(
        module: &naga::Module,
        structs_filter: Option<HashSet<String>>,
        args: &ModuleToTokensConfig,
    ) -> Self {
        let mut res = Self {
            items: Vec::new(),
            references: HashMap::new(),
            structs_filter,
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

                let mut layouter = Layouter::default();
                layouter.update(module.to_ctx()).unwrap();

                let members_have_names = members.iter().all(|member| member.name.is_some());
                let mut last_field_name = None;
                let mut total_offset = 0;
                let mut largest_alignment = 0;

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
                                let alignment = layouter[member.ty].alignment * 1u32;
                                largest_alignment = largest_alignment.max(alignment);
                                let padding_needed =
                                    layouter[member.ty].alignment.round_up(total_offset)
                                        - total_offset;
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
                if !struct_alignment.is_aligned(total_offset) {
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
