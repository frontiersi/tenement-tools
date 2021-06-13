from traitlets import (HasTraits, List, Unicode, Dict, Type, CaselessStrEnum)
from jupyter_client.kernelspec import KernelSpecManager

class ProKernelSpec(HasTraits):
    argv = List()
    display_name = 'ArcGISPro'
    language = 'Python'
    env = Dict()
    resource_dir = Unicode()
    interrupt_mode = CaselessStrEnum(
        ['message', 'signal'], default_value='signal'
    )
    metadata = Dict()

    @classmethod
    def from_resource_dir(cls, resource_dir):
        kernel_dict = { 'argv': [], 'display_name': 'ArcGISPro', 'language': 'python', }
        return cls(resource_dir=resource_dir, **kernel_dict)

    def to_dict(self):
        d = dict(argv=self.argv,
                 env=self.env,
                 display_name=self.display_name,
                 language=self.language,
                 interrupt_mode=self.interrupt_mode,
                 metadata=self.metadata,
                )

        return d

class ProKernelSpecManager(KernelSpecManager):

    kernel_spec_class = Type(ProKernelSpec, config=True,
        help="""The kernel spec class.  This is configurable to allow
        subclassing of the KernelSpecManager for customized behavior.
        """
    )
