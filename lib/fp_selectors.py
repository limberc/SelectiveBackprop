class PrimedSelector:
    def __init__(self, selector_type, initial_num_images, staleness, epoch=0):
        self.initial = AlwaysOnSelector()
        assert selector_type in ['alwayson', 'stale'], "FP Selector must be in {alwayson, stale}"
        self.final = AlwaysOnSelector() if selector_type == 'alwayson' else StaleSelector(staleness)
        self.initial_num_images = initial_num_images
        self.num_trained = 0

    def next_partition(self, partition_size):
        self.num_trained += partition_size

    def get_selector(self):
        return self.initial if self.num_trained < self.initial_num_images else self.final

    def select(self, *args, **kwargs):
        return self.get_selector().select(*args, **kwargs)

    def mark(self, *args, **kwargs):
        return self.get_selector().mark(*args, **kwargs)


class AlwaysOnSelector:
    def mark(self, examples_and_metadata):
        for em in examples_and_metadata:
            em.example.forward_select_probability = 1.
            em.example.forward_select = True
        return examples_and_metadata


class StaleSelector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.logger = {"counter": 0, "forward": 0, "no_forward": 0}

    def select(self, em):
        self.logger['counter'] += 1

        em.metadata["epochs_since_update"] += 1
        if 'loss' not in em.metadata or em.metadata["epochs_since_update"] >= self.threshold:
            self.logger['forward'] += 1
            return True
        else:
            self.logger['no_forward'] += 1
            em.example.loss = em.metadata["loss"]
            return False

    def mark(self, examples_and_metadata):
        for em in examples_and_metadata:
            em.example.forward_select = self.select(em)
        return examples_and_metadata
