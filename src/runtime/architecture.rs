/// Semantic model identity used to bind loaders and runners to typed task I/O.
pub trait ModelArchitecture {
    type Input<'a>;
    type Output;
}
