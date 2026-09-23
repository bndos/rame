use crate::tensor::Device;

#[derive(Debug, Clone)]
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

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
        }
    }
}
