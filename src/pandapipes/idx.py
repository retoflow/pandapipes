

class IndexMeta(type):
    def __new__(cls, name, bases, classdict):
        clsobj = super().__new__(cls, name, bases, classdict)

        # Sammle alle int-Attribute (außer __dunder__)
        clsobj._indices = {
            k: v for k, v in classdict.items()
            if not k.startswith("__") and isinstance(v, int)
        }

        # Vererbung berücksichtigen
        for base in bases:
            if hasattr(base, "_indices"):
                clsobj._indices = {**base._indices, **clsobj._indices}

        return clsobj

    def __iter__(cls):
        return iter(cls._indices.items())

    def keys(cls):
        return cls._indices.keys()

    def values(cls):
        return cls._indices.values()

    def items(cls):
        return cls._indices.items()