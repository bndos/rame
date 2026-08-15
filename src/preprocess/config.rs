use crate::tensor::Device;

#[derive(Debug, Clone, Default)]
pub struct PreprocessConfig {
    pub device: Device,
}

impl PreprocessConfig {
    pub fn new(device: impl Into<Device>) -> Self {
        Self {
            device: device.into(),
        }
    }
}
