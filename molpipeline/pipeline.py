"""Defines the pipeline which handles pipeline elements."""
import multiprocessing
from typing import Any, Iterable, Optional

from molpipeline.pipeline_elements.abstract_pipeline_elements import AnyPipeElement
from molpipeline.utils.multi_proc import check_available_cores


class MolPipeline:
    _n_jobs: int
    _pipeline_element_list: list[AnyPipeElement]

    def __init__(self, pipeline_element_list: list[AnyPipeElement], n_jobs: int = 1):
        self._pipeline_element_list = pipeline_element_list
        self.n_jobs = n_jobs

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, requested_jobs: int) -> None:
        self._n_jobs = check_available_cores(requested_jobs)

    def finish(self) -> None:
        for p_element in self._pipeline_element_list:  # type: AnyPipeElement
            p_element.finish()

    def fit(
            self,
            x_input: Any,
            y_input: Any = None,
            **fit_params: dict[Any, Any],
            ) -> None:
        self.fit_transform(x_input)

    def fit_transform(
            self,
            x_input: Any,
            y_input: Any = None,
            **fit_params: dict[str, Any],
    ) -> Any:

        iter_input = x_input
        for p_element in self._pipeline_element_list:
            iter_input = p_element.fit_transform(iter_input)  # TODO: Parallel processing
        return iter_input

    def transform_single(self, input_value: Any) -> Any:
        iter_value = input_value
        for p_element in self._pipeline_element_list:  # type: AnyPipeElement
            iter_value = p_element.transform_single(iter_value)
        return iter_value

    def transform(self, x_input: Any) -> Any:
        last_element = self._pipeline_element_list[-1]
        if hasattr(last_element, "collect_singles"):
            return last_element.collect_singles((single for single in self._transform_iterator(x_input)))
        else:
            return list(self._transform_iterator(x_input))

    def _transform_iterator(self, x_input: Any) -> Any:
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                for transformed_value in pool.imap(self.transform_single, x_input):
                    yield transformed_value
        else:
            for value in x_input:
                yield self.transform_single(value)
        self.finish()
