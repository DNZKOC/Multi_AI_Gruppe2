from cms.plugin_base import CMSPluginBase
from cms.plugin_pool import plugin_pool
from django.utils.translation import ugettext_lazy as _
from download.models import Download


class CMSPluginDownload(CMSPluginBase):
    model = Download
    name = _("Download")
    render_template = "download/plugin.html"
    module = _("TheHerk")

    def render(self, context, instance, placeholder):
        context['instance'] = instance
        return context

plugin_pool.register_plugin(CMSPluginDownload)
