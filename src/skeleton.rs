use std::io::{Read, Seek, SeekFrom};

use anyhow::{bail, Result};
use binrw::BinReaderExt;
use nalgebra::{Matrix4, Vector3};

const MODEL_MAGIC: u32 = 0xFFFF0003;

pub type Mat4 = Matrix4<f32>;
pub type Vec3 = Vector3<f32>;

#[derive(Debug, Clone)]
pub struct Skeleton {
    pub character_id: i16,
    hierarchy: Vec<Option<usize>>,
    default_transforms: Vec<Mat4>,
}

impl Skeleton {
    pub const fn from_hierarchy(character_id: i16, hierarchy: Vec<Option<usize>>) -> Self {
        Self {
            character_id,
            hierarchy,
            default_transforms: Vec::new(),
        }
    }

    pub fn read_from_model(mut f: impl Read + Seek) -> Result<Self> {
        f.seek(SeekFrom::Start(4))?;

        let character_id: i32 = f.read_le()?;
        f.seek(SeekFrom::Current(12))?;
        let data_offset: u32 = f.read_le()?;

        f.seek(SeekFrom::Start(data_offset as u64))?;

        let magic: u32 = f.read_le()?;
        if magic != MODEL_MAGIC {
            bail!("Invalid model magic number: {:#08X}", magic);
        }
        f.seek(SeekFrom::Current(4))?;
        let transforms_offset: u32 = f.read_le()?;
        let bone_count: u32 = f.read_le()?;
        let hierarchy_offset: u32 = f.read_le()?;

        f.seek(SeekFrom::Start((data_offset + transforms_offset) as u64))?;
        let mut transforms = Vec::with_capacity(bone_count as usize);
        for _ in 0..bone_count {
            let floats: [f32; 16] = f.read_le()?;
            // D3D-style row-major matrices need to be transposed for nalgebra, but because
            // from_iterator expects values in column-major order, this happens automatically
            transforms.push(Mat4::from_column_slice(&floats[..]));
        }

        f.seek(SeekFrom::Start((data_offset + hierarchy_offset) as u64))?;
        let mut hierarchy = Vec::with_capacity(bone_count as usize);
        for _ in 0..bone_count {
            let bone_id: u8 = f.read_le()?;
            hierarchy.push((bone_id != 0xFF).then(|| bone_id as usize));
        }

        Ok(Self {
            character_id: character_id as i16,
            hierarchy,
            default_transforms: transforms,
        })
    }

    pub fn iter_hierarchy(&self) -> impl Iterator<Item = Option<usize>> + use<'_> {
        self.hierarchy.iter().copied()
    }

    pub fn num_bones(&self) -> usize {
        self.hierarchy.len()
    }

    pub fn get_transform_relative_to(&self, bone_id: usize, parent_id: usize) -> Mat4 {
        if self.default_transforms.is_empty() {
            return Mat4::identity();
        }

        let transform = self.default_transforms[bone_id].clone();
        let parent_transform = &self.default_transforms[parent_id];
        let parent_inverse = parent_transform.try_inverse().unwrap_or_default();
        parent_inverse * transform
    }

    pub fn get_relative_transform(&self, bone_id: usize) -> Mat4 {
        match self.hierarchy[bone_id] {
            Some(parent_id) => self.get_transform_relative_to(bone_id, parent_id),
            None => self.default_transforms.get(bone_id).cloned().unwrap_or_default(),
        }
    }

    pub fn get_translation_lengths(&self) -> Vec<f32> {
        let mut translation_lengths = Vec::with_capacity(self.num_bones());
        for i in 0..self.num_bones() {
            let transform = self.get_relative_transform(i);
            let translation = Vec3::new(transform.m14, transform.m24, transform.m34);
            translation_lengths.push(translation.norm());
        }

        translation_lengths
    }
}