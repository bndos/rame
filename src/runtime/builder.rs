use crate::RameResult;
use crate::runtime::{ModelArchitecture, ModelLoader};
use crate::sources::ResolveModelSource;

#[derive(Debug, Clone, Copy)]
pub struct Missing;

/// Collects source and artifact choices before loading a typed model.
#[derive(Debug, Clone)]
pub struct ModelBuilder<M, S = Missing, A = Missing> {
    architecture: M,
    source: S,
    artifact: A,
}

pub type BuiltModel<M, A> = <A as ModelLoader<M>>::Runner;

impl<M> ModelBuilder<M> {
    pub fn new(architecture: M) -> Self {
        Self {
            architecture,
            source: Missing,
            artifact: Missing,
        }
    }
}

impl<M, S, A> ModelBuilder<M, S, A> {
    pub fn source<NextSource>(self, source: NextSource) -> ModelBuilder<M, NextSource, A> {
        ModelBuilder {
            architecture: self.architecture,
            source,
            artifact: self.artifact,
        }
    }

    pub fn artifact<NextArtifact>(
        self,
        artifact: NextArtifact,
    ) -> ModelBuilder<M, S, NextArtifact> {
        ModelBuilder {
            architecture: self.architecture,
            source: self.source,
            artifact,
        }
    }
}

impl<M, S, A> ModelBuilder<M, S, A>
where
    M: ModelArchitecture,
    S: ResolveModelSource,
    A: ModelLoader<M>,
{
    pub fn build(self) -> RameResult<BuiltModel<M, A>> {
        let source = self.source.resolve_model_source()?;
        self.artifact.load(self.architecture, source)
    }
}

#[cfg(test)]
mod tests {
    use crate::RameResult;
    use crate::runtime::{ModelArchitecture, ModelLoader, ModelRunner};
    use crate::sources::ResolvedModelSource;

    use super::ModelBuilder;

    #[derive(Debug, Clone, Copy)]
    struct TestArchitecture;

    impl ModelArchitecture for TestArchitecture {
        type Input<'a> = i32;
        type Output = i32;
    }

    struct RepeatedRunner {
        iterations: usize,
    }

    impl ModelRunner for RepeatedRunner {
        type Architecture = TestArchitecture;

        fn run_many(&mut self, inputs: &[i32]) -> RameResult<Vec<i32>> {
            let mut outputs = inputs.to_vec();
            for _ in 0..self.iterations {
                for output in &mut outputs {
                    *output += 1;
                }
            }
            Ok(outputs)
        }
    }

    struct RepeatedLoader {
        iterations: usize,
    }

    impl ModelLoader<TestArchitecture> for RepeatedLoader {
        type Runner = RepeatedRunner;

        fn load(
            self,
            _architecture: TestArchitecture,
            _source: ResolvedModelSource,
        ) -> RameResult<Self::Runner> {
            Ok(RepeatedRunner {
                iterations: self.iterations,
            })
        }
    }

    #[test]
    fn builds_a_custom_runner_with_repeated_control_flow() {
        let mut runner = ModelBuilder::new(TestArchitecture)
            .source(".")
            .artifact(RepeatedLoader { iterations: 3 })
            .build()
            .unwrap();

        assert_eq!(runner.run_many(&[1, 2]).unwrap(), vec![4, 5]);
    }
}
