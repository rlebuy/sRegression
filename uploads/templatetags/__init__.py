import os
from django import template

register = template.Library()

@register.filter
def basename(value):
    """Devuelve sólo el último componente del path, con extensión."""
    return os.path.basename(value)