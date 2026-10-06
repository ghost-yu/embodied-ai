from abc import abstractmethod


class BaseConnector:

    @abstractmethod
    def get_name(self):
        return "base"

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def generator(self, instance_id, session_id, n_steps_max):
        pass
