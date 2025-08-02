from kfp import dsl
from kfp import compiler
from typing import NamedTuple


@dsl.component
def num_add(a: int, b: int) -> int:
    '''Calculates sum of two arguments'''
    return a + b


@dsl.component
def num_sub(a: int, b: int) -> int:
    '''Calculates difference of two arguments'''
    return a - b


@dsl.pipeline(name="number-pipeline")
def num_pipeline(a: int = 100, b: int = -100) -> NamedTuple('Outputs', [('final_result', int)]):
    add_task = num_add(a=a, b=b)
    sub_task = num_sub(a=100, b=200)
    completed_task = num_add(a=add_task.output, b=sub_task.output)

    # ✅ FIXED: Return only the output of the final task
    return {"final_result": completed_task.output}


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=num_pipeline,
        package_path="num_pipeline.yaml"
    )


