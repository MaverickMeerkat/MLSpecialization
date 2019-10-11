class Layer(object):
    def __init__(self, units, activation, keep_prob):
        self.units = units
        self.activation = activation
        self.keep_prob = keep_prob  # dropout


class Architecture(object):
    def __init__(self, layers):
        self._layers = layers

    @classmethod
    def from_list(cls, list):
        layers = []
        for (u, a, p) in list:
            layers.append(Layer(u, a, p))
        return cls(layers)

    def get_layers(self):
        return self._layers

    def append_layer(self, layer):
        self._layers.append(layer)