import os
from django import template

register = template.Library()

@register.filter
def basename(value):
    """
    Devuelve sólo el nombre de archivo (último componente del path),
    incluyendo extensión.
    """
    return os.path.basename(value)