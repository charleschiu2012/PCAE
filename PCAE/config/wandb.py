class WandbConfig:
    def __init__(self, project_name, run_name, dir_path, machine_id, step_loss_freq=0, visual_flag=False):
        self._project_name = project_name
        self._run_name = run_name
        self.dir_path = dir_path
        self.machine_id = machine_id
        self.step_loss_freq = step_loss_freq
        self.visual_flag = visual_flag

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, new_project_name):
        assert isinstance(new_project_name, str)
        self._project_name = new_project_name

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, new_run_name):
        assert isinstance(new_run_name, str)
        self._run_name = new_run_name
