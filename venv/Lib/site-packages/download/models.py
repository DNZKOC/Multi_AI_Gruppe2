from cms.models.pluginmodel import CMSPlugin
from django.db import models


class Download(CMSPlugin):
    name = models.CharField(max_length=256)
    url = models.URLField()
    icon = models.CharField(
        max_length=24,
        default="icon-download-alt",
        blank=True
    )
