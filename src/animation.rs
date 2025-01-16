use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::Result;
use binrw::{binrw, BinRead, BinResult, BinWrite, Endian};
use half::prelude::*;
use nalgebra::{Matrix3, Rotation3, Vector3};

use crate::skeleton::Skeleton;

type Mat3 = Matrix3<f32>;
type Rot3 = Rotation3<f32>;
type Vec3 = Vector3<f32>;

// wrapper types because we can't implement BinRead/BinWrite for f16
#[binrw]
#[derive(Debug, Clone, Copy)]
struct Float16(u16);

impl Float16 {
    pub fn to_f32(&self) -> f32 {
        f16::from_bits(self.0).to_f32()
    }

    pub fn from_f32(f: f32) -> Self {
        Self(f16::from_f32(f).to_bits())
    }
}

#[binrw]
#[derive(Debug, Clone, Copy)]
struct Float16Padded(u16, u16);

impl Float16Padded {
    pub fn to_f32(&self) -> f32 {
        f16::from_bits(self.0).to_f32()
    }

    pub fn from_f32(f: f32) -> Self {
        Self(f16::from_f32(f).to_bits(), 0)
    }
}

impl From<Float16> for Float16Padded {
    fn from(value: Float16) -> Self {
        Self(value.0, 0)
    }
}

impl From<Float16Padded> for Float16 {
    fn from(value: Float16Padded) -> Self {
        Self(value.0)
    }
}

#[binrw]
#[derive(Debug, Clone)]
struct Vec3Fixed12 {
    x: i16,
    y: i16,
    z: i16,
}

impl Vec3Fixed12 {
    pub fn to_rotation(&self) -> Rot3 {
        Rot3::from_euler_angles((self.x as f32) / 4096.0, (self.y as f32) / 4096.0, (self.z as f32) / 4096.0)
    }
}

impl From<&Rot3> for Vec3Fixed12 {
    fn from(value: &Rot3) -> Self {
        let (x, y, z) = value.euler_angles();
        Self {
            x: (x * 4096.0) as i16,
            y: (y * 4096.0) as i16,
            z: (z * 4096.0) as i16,
        }
    }
}

impl From<Rot3> for Vec3Fixed12 {
    fn from(value: Rot3) -> Self {
        (&value).into()
    }
}

#[binrw]
#[derive(Debug, Clone)]
struct Vec3Float16 {
    x: Float16,
    y: Float16,
    z: Float16,
}

impl Vec3Float16 {
    pub fn vec3(&self) -> Vec3 {
        Vec3::new(self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }
}

impl From<&Vec3> for Vec3Float16 {
    fn from(value: &Vec3) -> Self {
        Self {
            x: Float16::from_f32(value.x),
            y: Float16::from_f32(value.y),
            z: Float16::from_f32(value.z),
        }
    }
}

impl From<Vec3> for Vec3Float16 {
    fn from(value: Vec3) -> Self {
        (&value).into()
    }
}

impl From<&Vec3Float16Padded> for Vec3Float16 {
    fn from(value: &Vec3Float16Padded) -> Self {
        Self {
            x: value.x.into(),
            y: value.y.into(),
            z: value.z.into(),
        }
    }
}

impl From<&Vec3Float32> for Vec3Float16 {
    fn from(value: &Vec3Float32) -> Self {
        Self {
            x: Float16::from_f32(value.x),
            y: Float16::from_f32(value.y),
            z: Float16::from_f32(value.z),
        }
    }
}

#[binrw]
#[derive(Debug, Clone)]
struct Vec3Float16Padded {
    x: Float16Padded,
    y: Float16Padded,
    z: Float16Padded,
}

impl Vec3Float16Padded {
    pub fn vec3(&self) -> Vec3 {
        Vec3::new(self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }
}

impl From<&Vec3> for Vec3Float16Padded {
    fn from(value: &Vec3) -> Self {
        Self {
            x: Float16Padded::from_f32(value.x),
            y: Float16Padded::from_f32(value.y),
            z: Float16Padded::from_f32(value.z),
        }
    }
}

impl From<Vec3> for Vec3Float16Padded {
    fn from(value: Vec3) -> Self {
        (&value).into()
    }
}

impl From<&Vec3Float16> for Vec3Float16Padded {
    fn from(value: &Vec3Float16) -> Self {
        Self {
            x: value.x.into(),
            y: value.y.into(),
            z: value.z.into(),
        }
    }
}

impl From<&Vec3Float32> for Vec3Float16Padded {
    fn from(value: &Vec3Float32) -> Self {
        Self {
            x: Float16Padded::from_f32(value.x),
            y: Float16Padded::from_f32(value.y),
            z: Float16Padded::from_f32(value.z),
        }
    }
}

#[binrw]
#[derive(Debug, Clone)]
struct Vec3Float32 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3Float32 {
    pub fn vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl From<&Vec3> for Vec3Float32 {
    fn from(value: &Vec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Vec3> for Vec3Float32 {
    fn from(value: Vec3) -> Self {
        (&value).into()
    }
}

impl From<&Vec3Float16> for Vec3Float32 {
    fn from(value: &Vec3Float16) -> Self {
        Self {
            x: value.x.to_f32(),
            y: value.y.to_f32(),
            z: value.z.to_f32(),
        }
    }
}

impl From<&Vec3Float16Padded> for Vec3Float32 {
    fn from(value: &Vec3Float16Padded) -> Self {
        Self {
            x: value.x.to_f32(),
            y: value.y.to_f32(),
            z: value.z.to_f32(),
        }
    }
}

#[binrw]
#[derive(Debug, Clone)]
struct Vec4Unit16 {
    x: i16,
    y: i16,
    z: i16,
    w: i16,
}

#[binrw]
#[derive(Debug, Clone)]
#[br(import(id: u8, is_root: bool))]
enum Transform {
    #[br(pre_assert(id == 0))]
    None,
    #[br(pre_assert(id == 1))]
    RotationFixed12(Vec3Fixed12),
    #[br(pre_assert(id == 2 && is_root))]
    IsometryFloat32 {
        translation: Vec3Float32,
        rotation: Vec3Fixed12,
    },
    #[br(pre_assert(id == 2 && !is_root))]
    IsometryFloat16 {
        translation: Vec3Float16,
        rotation: Vec3Fixed12,
    },
    #[br(pre_assert(id == 3))]
    RotationWithAxis {
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 4 && !is_root))]
    IsometryWithAxisFloat16 {
        translation: Vec3Float16,
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 4 && is_root))]
    IsometryWithAxisFloat16Padded {
        translation: Vec3Float16Padded,
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 5 && is_root))]
    Isometry5WithAxisFloat32 {
        translation: Vec3Float32,
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 5 && !is_root))]
    Isometry5WithAxisFloat16 {
        translation: Vec3Float16,
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 6 && !is_root))]
    FullDynamicTransformFloat16Padded {
        translation_start: Vec3Float16, // only the second translation vector is padded
        translation_end: Vec3Float16Padded,
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 6 && is_root))]
    FullDynamicTransformFloat32 {
        translation_start: Vec3Float32,
        translation_end: Vec3Float32,
        rotation: Vec3Fixed12,
        axis: Vec4Unit16,
        angle: i16,
    },
    #[br(pre_assert(id == 7))]
    Empty,
}

impl Transform {
    pub const fn id(&self) -> u8 {
        match self {
            Self::None => 0,
            Self::RotationFixed12(_) => 1,
            Self::IsometryFloat32 { .. } | Self::IsometryFloat16 { .. } => 2,
            Self::RotationWithAxis { .. } => 3,
            Self::IsometryWithAxisFloat16 { .. } | Self::IsometryWithAxisFloat16Padded { .. } => 4,
            Self::Isometry5WithAxisFloat32 { .. } | Self::Isometry5WithAxisFloat16 { .. } => 5,
            Self::FullDynamicTransformFloat32 { .. } | Self::FullDynamicTransformFloat16Padded { .. } => 6,
            Self::Empty => 7,
        }
    }

    pub fn rotation(&self) -> Rot3 {
        match self {
            Self::RotationFixed12(rotation) => rotation.to_rotation(),
            Self::IsometryFloat32 { rotation, .. } | Self::IsometryFloat16 { rotation, .. } => rotation.to_rotation(),
            Self::RotationWithAxis { rotation, .. } => rotation.to_rotation(),
            Self::IsometryWithAxisFloat16 { rotation, .. } | Self::IsometryWithAxisFloat16Padded { rotation, .. } => rotation.to_rotation(),
            Self::Isometry5WithAxisFloat32 { rotation, .. } | Self::Isometry5WithAxisFloat16 { rotation, .. } => rotation.to_rotation(),
            Self::FullDynamicTransformFloat32 { rotation, .. } | Self::FullDynamicTransformFloat16Padded { rotation, .. } => rotation.to_rotation(),
            _ => Rot3::identity(),
        }
    }

    pub fn translation_start(&self) -> Vec3 {
        match self {
            Self::IsometryFloat32 { translation, .. } | Self::Isometry5WithAxisFloat32 { translation, .. } => translation.vec3(),
            Self::IsometryFloat16 { translation, .. } | Self::IsometryWithAxisFloat16 { translation, .. } | Self::Isometry5WithAxisFloat16 { translation, .. } => translation.vec3(),
            Self::IsometryWithAxisFloat16Padded { translation, .. } => translation.vec3(),
            Self::FullDynamicTransformFloat32 { translation_start, .. } => translation_start.vec3(),
            Self::FullDynamicTransformFloat16Padded { translation_start, .. } => translation_start.vec3(),
            _ => Vec3::default(),
        }
    }

    pub fn translation_end(&self) -> Vec3 {
        match self {
            Self::IsometryFloat32 { translation, .. } | Self::Isometry5WithAxisFloat32 { translation, .. } => translation.vec3(),
            Self::IsometryFloat16 { translation, .. } | Self::IsometryWithAxisFloat16 { translation, .. } | Self::Isometry5WithAxisFloat16 { translation, .. } => translation.vec3(),
            Self::IsometryWithAxisFloat16Padded { translation, .. } => translation.vec3(),
            Self::FullDynamicTransformFloat32 { translation_end, .. } => translation_end.vec3(),
            Self::FullDynamicTransformFloat16Padded { translation_end, .. } => translation_end.vec3(),
            _ => Vec3::default(),
        }
    }

    pub fn set_transform(&mut self, new_rotation: &Rot3, new_translation_start: &Vec3, new_translation_end: &Vec3) {
        match self {
            Self::RotationFixed12(_) => {
                *self = Self::RotationFixed12(new_rotation.into());
            }
            Self::IsometryFloat32 { rotation, translation } => {
                *rotation = new_rotation.into();
                *translation = new_translation_end.into();
            }
            Self::IsometryFloat16 { rotation, translation } => {
                *rotation = new_rotation.into();
                *translation = new_translation_end.into();
            }
            Self::RotationWithAxis { rotation, .. } => {
                *rotation = new_rotation.into();
            }
            Self::IsometryWithAxisFloat16 { rotation, translation, .. } | Self::Isometry5WithAxisFloat16 { rotation, translation, .. } => {
                *rotation = new_rotation.into();
                *translation = new_translation_end.into();
            }
            Self::IsometryWithAxisFloat16Padded { rotation, translation, .. } => {
                *rotation = new_rotation.into();
                *translation = new_translation_end.into();
            }
            Self::Isometry5WithAxisFloat32 { rotation, translation, ..} => {
                *rotation = new_rotation.into();
                *translation = new_translation_end.into();
            }
            Self::FullDynamicTransformFloat16Padded { rotation, translation_start, translation_end, .. } => {
                *rotation = new_rotation.into();
                *translation_start = new_translation_start.into();
                *translation_end = new_translation_end.into();
            }
            Self::FullDynamicTransformFloat32 { rotation, translation_start, translation_end, .. } => {
                *rotation = new_rotation.into();
                *translation_start = new_translation_start.into();
                *translation_end = new_translation_end.into();
            }
            _ => (),
        }
    }

    pub fn rotate(&mut self, rotation: &Rot3) {
        if self.is_empty() {
            return;
        }

        // FIXME: this doesn't handle interpolated rotations
        // however, I don't think axis and angle (and I'm only guessing that that's what they are)
        // are actually used; the game seems to recalculate these without ever looking at the values
        // in the file, which means they may not be accurate. replicating the game's logic for the
        // recalculation would be very cumbersome, and as far as I've seen, player animations don't
        // use these transform types anyway, so I'm not going to bother for now.

        let new_rotation = rotation * self.rotation();
        let new_translation_start = rotation.transform_vector(&self.translation_start());
        let new_translation_end = rotation.transform_vector(&self.translation_end());

        self.set_transform(&new_rotation, &new_translation_start, &new_translation_end);
    }

    pub fn set_translation(&mut self, new_translation_start: &Vec3, new_translation_end: &Vec3) {
        match self {
            Self::IsometryFloat32 { translation, .. } => {
                *translation = new_translation_end.into();
            }
            Self::IsometryFloat16 { translation, .. } => {
                *translation = new_translation_end.into();
            }
            Self::IsometryWithAxisFloat16 { translation, .. } | Self::Isometry5WithAxisFloat16 { translation, .. } => {
                *translation = new_translation_end.into();
            }
            Self::IsometryWithAxisFloat16Padded { translation, .. } => {
                *translation = new_translation_end.into();
            }
            Self::Isometry5WithAxisFloat32 { translation, ..} => {
                *translation = new_translation_end.into();
            }
            Self::FullDynamicTransformFloat16Padded { translation_start, translation_end, .. } => {
                *translation_start = new_translation_start.into();
                *translation_end = new_translation_end.into();
            }
            Self::FullDynamicTransformFloat32 { translation_start, translation_end, .. } => {
                *translation_start = new_translation_start.into();
                *translation_end = new_translation_end.into();
            }
            _ => (),
        }
    }

    pub fn translate(&mut self, translation_start: &Vec3, translation_end: &Vec3) {
        if self.is_empty() {
            return;
        }

        let rotation = self.rotation();
        let new_translation_start = self.translation_start() + (rotation * translation_start);
        let new_translation_end = self.translation_end() + (rotation * translation_end);

        self.set_translation(&new_translation_start, &new_translation_end);
    }

    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::None | Self::Empty)
    }

    /*pub const fn is_root(&self) -> bool {
        matches!(self, Self::IsometryFloat32 { .. } | Self::Isometry5WithAxisFloat32 { .. } | Self::FullDynamicTransformFloat32 { .. } | Self::IsometryWithAxisFloat16Padded { .. })
    }*/

    pub fn to_root(&self) -> Self {
        match self {
            Self::IsometryFloat16 { translation, rotation } => {
                Self::IsometryFloat32 {
                    translation: translation.into(),
                    rotation: rotation.clone(),
                }
            }
            Self::IsometryWithAxisFloat16 { translation, rotation, axis, angle } => {
                Self::IsometryWithAxisFloat16Padded {
                    translation: translation.into(),
                    rotation: rotation.clone(),
                    axis: axis.clone(),
                    angle: *angle,
                }
            }
            Self::Isometry5WithAxisFloat16 { translation, rotation, axis, angle } => {
                Self::Isometry5WithAxisFloat32 {
                    translation: translation.into(),
                    rotation: rotation.clone(),
                    axis: axis.clone(),
                    angle: *angle,
                }
            }
            Self::FullDynamicTransformFloat16Padded { translation_start, translation_end, rotation, axis, angle } => {
                Self::FullDynamicTransformFloat32 {
                    translation_start: translation_start.into(),
                    translation_end: translation_end.into(),
                    rotation: rotation.clone(),
                    axis: axis.clone(),
                    angle: *angle,
                }
            }
            _ => self.clone(),
        }
    }

    pub fn to_child(&self) -> Self {
        match self {
            Self::IsometryFloat32 { translation, rotation } => {
                Self::IsometryFloat16 {
                    translation: translation.into(),
                    rotation: rotation.clone(),
                }
            }
            Self::IsometryWithAxisFloat16Padded { translation, rotation, axis, angle } => {
                Self::IsometryWithAxisFloat16 {
                    translation: translation.into(),
                    rotation: rotation.clone(),
                    axis: axis.clone(),
                    angle: *angle,
                }
            }
            Self::Isometry5WithAxisFloat32 { translation, rotation, axis, angle } => {
                Self::Isometry5WithAxisFloat16 {
                    translation: translation.into(),
                    rotation: rotation.clone(),
                    axis: axis.clone(),
                    angle: *angle,
                }
            }
            Self::FullDynamicTransformFloat32 { translation_start, translation_end, rotation, axis, angle } => {
                Self::FullDynamicTransformFloat16Padded {
                    translation_start: translation_start.into(),
                    translation_end: translation_end.into(),
                    rotation: rotation.clone(),
                    axis: axis.clone(),
                    angle: *angle,
                }
            }
            _ => self.clone(),
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::None
    }
}

const BLOCK_SIZE: usize = 8;

#[derive(Debug, Clone)]
struct TransformBlock {
    transforms: [(bool, Transform); BLOCK_SIZE],
}

impl TransformBlock {
    pub fn from_slice(transforms: &[(bool, Transform)]) -> Self {
        let mut block = Self::default();
        block.transforms.clone_from_slice(transforms);

        block
    }

    pub const fn header(&self) -> u32 {
        let mut header = 0u32;

        let mut i = 0;
        while i < self.transforms.len() {
            let (flag, transform) = &self.transforms[i];
            header |= ((transform.id() | if *flag { 8 } else { 0 }) as u32) << (i * 4);

            i += 1;
        }

        header
    }
}

impl Default for TransformBlock {
    fn default() -> Self {
        Self {
            transforms: [
                (false, Transform::default()),
                (false, Transform::default()),
                (false, Transform::default()),
                (false, Transform::default()),
                (false, Transform::default()),
                (false, Transform::default()),
                (false, Transform::default()),
                (false, Transform::default()),
            ],
        }
    }
}

impl BinRead for TransformBlock {
    type Args<'a> = &'a [bool; BLOCK_SIZE];

    fn read_options<R: Read + Seek>(reader: &mut R, endian: Endian, args: Self::Args<'_>) -> BinResult<Self> {
        let mut header = u32::read_options(reader, endian, ())?;

        let mut block = Self::default();
        for ((flag, transform), &is_root) in block.transforms.iter_mut().zip(args) {
            let id = (header & 7) as u8;
            *flag = header & 8 != 0;
            header >>= 4;

            *transform = Transform::read_options(reader, endian, (id, is_root))?;
        }

        Ok(block)
    }
}

impl BinWrite for TransformBlock {
    type Args<'a> = ();

    fn write_options<W: Write + Seek>(&self, writer: &mut W, endian: Endian, _: Self::Args<'_>) -> BinResult<()> {
        self.header().write_options(writer, endian, ())?;
        for (_, transform) in &self.transforms {
            transform.write_options(writer, endian, ())?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct Animation {
    skeleton: Vec<Option<usize>>,
    skeleton_translation_lengths: Vec<f32>,
    frames: Vec<Vec<(bool, Transform)>>,
}

impl Animation {
    pub fn read(mut f: impl Read + Seek, skeleton: &Skeleton) -> Result<Self> {
        let num_bones = skeleton.num_bones();
        let hierarchy: Vec<_> = skeleton.iter_hierarchy().collect();

        let mut frames = Vec::new();
        let mut frame = Vec::with_capacity(num_bones);

        let start = f.seek(SeekFrom::Current(0))?;
        let end = f.seek(SeekFrom::End(0))?;
        f.seek(SeekFrom::Start(start))?;

        while f.seek(SeekFrom::Current(0))? < end {
            let bone_index = frame.len();
            let mut root_flags = [false; BLOCK_SIZE];
            for i in bone_index..bone_index + BLOCK_SIZE {
                root_flags[i - bone_index] = hierarchy[i % num_bones].is_none();
            }

            let block = TransformBlock::read_le_args(&mut f, &root_flags)?;
            for transform in block.transforms.into_iter() {
                if transform.1.is_none() {
                    continue;
                }

                frame.push(transform);
                if frame.len() == num_bones {
                    frames.push(frame);
                    frame = Vec::with_capacity(num_bones);
                }
            }
        }

        Ok(Self {
            skeleton: hierarchy,
            skeleton_translation_lengths: skeleton.get_translation_lengths(),
            frames,
        })
    }

    fn compose_transform(&self, index: usize, frame: &[(bool, Transform)], is_mapped: &[bool]) -> (Rot3, Vec3, Vec3) {
        let transform = &frame[index].1;
        let rotation = transform.rotation();
        let translation_start = transform.translation_start();
        let translation_end = transform.translation_end();

        if let Some(parent_index) = self.skeleton[index] {
            // only need to compose if the parent has not been mapped
            if is_mapped[parent_index] {
                (rotation, translation_start, translation_end)
            } else {
                let (parent_rotation, parent_translation_start, parent_translation_end) = self.compose_transform(parent_index, frame, is_mapped);
                let composed_rotation = rotation * parent_rotation;
                let composed_translation_start = parent_rotation.transform_vector(&translation_start) + parent_translation_start;
                let composed_translation_end = parent_rotation.transform_vector(&translation_end) + parent_translation_end;
                (composed_rotation, composed_translation_start, composed_translation_end)
            }
        } else {
            (rotation, translation_start, translation_end)
        }
    }

    pub fn write_for_skeleton(&self, skeleton: &Skeleton, mapping: &[Option<usize>], use_model_translations: bool, mut f: impl Write + Seek) -> Result<()> {
        let mut is_mapped = vec![false; self.skeleton.len()];
        for mapped_index in mapping {
            if let Some(index) = mapped_index {
                is_mapped[*index] = true;
            }
        }

        let mapped_skeleton: Vec<_> = skeleton.iter_hierarchy().collect();
        let num_mapped_bones = skeleton.num_bones();
        let frame_size = num_mapped_bones.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;

        let mut default_translations = Vec::with_capacity(num_mapped_bones);
        if use_model_translations {
            for i in 0..num_mapped_bones {
                let transform = skeleton.get_relative_transform(i);
                default_translations.push(Vec3::new(transform.m14, transform.m24, transform.m34));
            }
        }

        for input_frame in &self.frames {
            let mut output_frame = Vec::with_capacity(frame_size);
            for (output_index, input_index) in mapping.iter().enumerate() {
                let is_root = mapped_skeleton[output_index].is_none();
                let (transform_flag, mut output_transform) = if let Some(input_index) = input_index {
                    let (transform_flag, mut input_transform) = input_frame[*input_index].clone();
                    if is_root {
                        input_transform = input_transform.to_root();
                    }

                    let (start_scale, end_scale) = if use_model_translations {
                        // instead of using the translation from the input animation, which is likely
                        // not appropriate for the proportions of the output body, we'll use
                        let translation_start = input_transform.translation_start();
                        let translation_end = input_transform.translation_end();

                        let reference_length = self.skeleton_translation_lengths[*input_index];
                        (translation_start.norm() / reference_length, translation_end.norm() / reference_length)
                    } else {
                        (1.0, 1.0)
                    };

                    if let Some(input_parent) = self.skeleton[*input_index] {
                        if !is_mapped[input_parent] {
                            // the input bone's parent isn't mapped. compose the transformations
                            // between it and its highest mapped parent.
                            let (parent_rotation, parent_translation_start, parent_translation_end) = self.compose_transform(input_parent, input_frame.as_slice(), is_mapped.as_slice());
                            input_transform.rotate(&parent_rotation);
                            input_transform.translate(&parent_translation_start, &parent_translation_end);
                        }
                    }

                    if use_model_translations {
                        let default_translation = &default_translations[output_index];
                        input_transform.set_translation(&(default_translation * start_scale), &(default_translation * end_scale));
                    }

                    (transform_flag, input_transform)
                } else {
                    // we don't have a mapping for this bone in the input skeleton. use the default
                    // transform from the output skeleton.
                    let default_transform = skeleton.get_relative_transform(output_index);
                    let rotation = Rot3::from_matrix(&Mat3::new(
                        default_transform.m11, default_transform.m12, default_transform.m13,
                        default_transform.m21, default_transform.m22, default_transform.m23,
                        default_transform.m31, default_transform.m32, default_transform.m33,
                    ));
                    let translation = Vec3::new(default_transform.m14, default_transform.m24, default_transform.m34);
                    // FIXME: need to pull the appropriate storage type from the reference animation,
                    //  because the game executable contains hard-coded frame sizes for each character
                    let new_transform = Transform::IsometryFloat32 {
                        rotation: rotation.into(),
                        translation: translation.into(),
                    };

                    (false, new_transform)
                };

                if !is_root {
                    output_transform = output_transform.to_child();
                }

                output_frame.push((transform_flag, output_transform));
            }

            // frame must be padded to a multiple of the block size
            let num_padding = frame_size - output_frame.len();
            for _ in 0..num_padding {
                output_frame.push((false, Transform::default()));
            }

            for block in output_frame.chunks(BLOCK_SIZE) {
                TransformBlock::from_slice(block).write_le(&mut f)?;
            }
        }

        Ok(())
    }
}