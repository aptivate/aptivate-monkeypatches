from django.template.context import Context
from django.test import TestCase

import aptivate_monkeypatches.monkeypatches

# from django.test.utils import override_settings
# @override_settings(CACHE_MIDDLEWARE_ALIAS='test',
#    CACHES={'test': {'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
#        'LOCATION': 'test_cache'}})


class MonkeypatchesTests(TestCase):
    def test_render_to_template_saves_context_without_breaking_caching(self):
        """
        We used to patch django.template.base.Template.render to poke the
        context into the response object, which is great for tests but it
        breaks caching when it tries to serialise the response, because the
        context can have all kinds of unserialisable objects in it.
        """

        # from django.test.client import RequestFactory
        # request = RequestFactory().get('/dummy')
        # request._cache_update_cache = True

        context = Context({
            'foo': 'bar',
            'func': lambda x: "whee",  # not serializable
        })

        def simple_view(request):
            from django.template import Template
            template = Template('hello world')

            from django.template.response import SimpleTemplateResponse
            return SimpleTemplateResponse(template, context)

        from django.conf.urls import patterns, url
        class TestUrls:
            urlpatterns = patterns('', url(r'test/$', simple_view, name='test'))

        from django.test.utils import override_settings
        with override_settings(
            ROOT_URLCONF=TestUrls,
            MIDDLEWARE_CLASSES=('django.middleware.cache.UpdateCacheMiddleware',),
            CACHE_MIDDLEWARE_ALIAS='test',
            CACHES={'test': {'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
                'LOCATION': 'test_cache'}}):

            from django.core.management.base import CommandError

            try:
                from django.core.management.commands.createcachetable import Command
                create_cache_table = Command()
                parser = create_cache_table.create_parser('foo', 'bar')
                options, args = parser.parse_args([])
                create_cache_table.execute('test_cache', **options.__dict__)
            except CommandError as e:
                # the table might already exist, which is not really an error!
                pass

            response = self.client.get('/test/')

            # from django.middleware.cache import UpdateCacheMiddleware
            # cache = UpdateCacheMiddleware()

        # cache.process_response(request, response)
        # response.render()

        # is the context still in the response? tests need this!
        self.assertIn('context', dir(response))
        self.assertEquals(list(context), list(response.context))
