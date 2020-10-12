class DatasetConfig:
    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 dataset_size: dict,
                 resample_amount: int,
                 not_train_class=None):
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        self._dataset_size = dataset_size
        self._split_dataset_types = ['train', 'test', 'valid']
        self._resample_amount = resample_amount
        self.not_train_class = not_train_class

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def split_dataset_size(self):
        return self._dataset_size

    @property
    def train_dataset_num(self) -> int:
        return self._dataset_size['train']

    @property
    def test_dataset_num(self) -> int:
        return self._dataset_size['test']

    @property
    def valid_dataset_num(self) -> int:
        return self._dataset_size['valid']

    @property
    def dataset_num(self) -> int:
        return self.train_dataset_num + self.test_dataset_num + self.valid_dataset_num

    @property
    def dataset_types(self):
        return self._split_dataset_types

    def get_dataset_num(self, dataset_type=None) -> int:
        if dataset_type is None:
            return self.dataset_num

        assert dataset_type in self._split_dataset_types
        return self._dataset_size[dataset_type]

    @property
    def resample_amount(self):
        return self._resample_amount
