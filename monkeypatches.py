from django.conf import LazySettings, global_settings
from django.contrib.admin.helpers import Fieldline, AdminField, mark_safe
from django.contrib.admin.views.main import ChangeList
from django.contrib.auth import models as auth_models
from django.core.urlresolvers import RegexURLResolver, NoReverseMatch
from django.db.models.fields import AutoField
from django.db.models.query import QuerySet
from django.forms import BaseForm
from django.forms.models import BaseModelForm, InlineForeignKeyField, \
    construct_instance, NON_FIELD_ERRORS
from django.template.response import TemplateResponse
from django.test.client import ClientHandler, RequestFactory, MULTIPART_CONTENT, \
    urlparse, FakePayload
from django.test.utils import ContextList
from aptivate_monkeypatch.monkeypatch import before, after, patch, insert
from pprint import PrettyPrinter
import django.template.loader

# import os
# os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'

@patch(ClientHandler, 'get_response') 
def get_response_with_exception_passthru(original_function, self, request):
    """
    Returns an HttpResponse object for the given HttpRequest. Unlike
    the original get_response, this does not catch exceptions, which
    allows you to see the full stack trace in your tests instead of
    a 500 error page.
    """
    
    # print("get_response(%s)" % request)
    
    from django.core import exceptions, urlresolvers
    from django.conf import settings

    # Setup default url resolver for this thread, this code is outside
    # the try/except so we don't get a spurious "unbound local
    # variable" exception in the event an exception is raised before
    # resolver is set
    urlconf = settings.ROOT_URLCONF
    urlresolvers.set_urlconf(urlconf)
    resolver = urlresolvers.RegexURLResolver(r'^/', urlconf)
    response = None
    # Apply request middleware
    for middleware_method in self._request_middleware:
        response = middleware_method(request)
        if response:
            break

    if response is None:
        if hasattr(request, "urlconf"):
            # Reset url resolver with a custom urlconf.
            urlconf = request.urlconf
            urlresolvers.set_urlconf(urlconf)
            resolver = urlresolvers.RegexURLResolver(r'^/', urlconf)

        callback, callback_args, callback_kwargs = resolver.resolve(
                request.path_info)

        # Apply view middleware
        for middleware_method in self._view_middleware:
            response = middleware_method(request, callback, callback_args, callback_kwargs)
            if response:
                break

    if response is None:
        try:
            response = callback(request, *callback_args, **callback_kwargs)
        except Exception, e:
            # If the view raised an exception, run it through exception
            # middleware, and if the exception middleware returns a
            # response, use that. Otherwise, reraise the exception.
            for middleware_method in self._exception_middleware:
                response = middleware_method(request, e)
                if response:
                    break
            if response is None:
                raise

    # Complain if the view returned None (a common error).
    if response is None:
        try:
            view_name = callback.func_name # If it's a function
        except AttributeError:
            view_name = callback.__class__.__name__ + '.__call__' # If it's a class
        raise ValueError("The view %s.%s didn't return an HttpResponse object." % (callback.__module__, view_name))

    # If the response supports deferred rendering, apply template
    # response middleware and the render the response
    if hasattr(response, 'render') and callable(response.render):
        for middleware_method in self._template_response_middleware:
            response = middleware_method(request, response)
        
        """
        try:
            response.render()
        except Exception as e:
            if isinstance(response, TemplateResponse):
                raise Exception("Failed to render template: %s: %s" %
                    (response.template_name, e))
        """

    # Reset URLconf for this thread on the way out for complete
    # isolation of request.urlconf
    urlresolvers.set_urlconf(None)

    # Apply response middleware, regardless of the response
    for middleware_method in self._response_middleware:
        response = middleware_method(request, response)
    response = self.apply_response_fixes(request, response)

    return response

def dont_apply_response_fixes(original_function, self, request, response):
    """
    It doesn't make any sense to rewrite location headers in tests,
    because the test client doesn't know or care what hostname is
    used in a request, so it could change in future without breaking
    most people's tests, EXCEPT tests for redirect URLs!
    """
    return response
# patch(ClientHandler, 'apply_response_fixes', dont_apply_response_fixes)

@patch(QuerySet, 'get')
def queryset_get_with_exception_detail(original_function, self, *args, **kwargs):
    """
    Performs the query and returns a single object matching the given
    keyword arguments. This version provides extra details about the query
    if it fails to find any results.
    """
    
    try:
        return original_function(self, *args, **kwargs)
    except self.model.DoesNotExist as e:
        raise self.model.DoesNotExist("%s (query was: %s, %s)" %
            (e, args, kwargs))

@patch(RequestFactory, 'post')
def post_with_string_data_support(original_function, self, path, data={},
    content_type=MULTIPART_CONTENT, **extra):
    """If the data doesn't have an items() method, then it's probably already
    been converted to a string (encoded), and if we try again we'll call
    the nonexistent items() method and fail, so just don't encode it at
    all."""
    if content_type == MULTIPART_CONTENT and getattr(data, 'items', None) is None:
        parsed = urlparse(path)
        r = {
            'CONTENT_LENGTH': len(data),
            'CONTENT_TYPE':   content_type,
            'PATH_INFO':      self._get_path(parsed),
            'QUERY_STRING':   parsed[4],
            'REQUEST_METHOD': 'POST',
            'wsgi.input':     FakePayload(data),
        }
        r.update(extra)
        return self.request(**r)
    else:
        return original_function(self, path, data, content_type, **extra)


@patch(BaseModelForm, '_post_clean')
def post_clean_with_simpler_validation(original_function, self):
    """
    Until https://code.djangoproject.com/ticket/16423#comment:3 is implemented,
    patch it in ourselves: do the same validation on objects when called
    from the form, as the object would do on itself.
    """
    
    opts = self._meta
    # Update the model instance with self.cleaned_data.
    # print "construct_instance with password = %s" % self.cleaned_data.get('password')
    self.instance = construct_instance(self, self.instance, opts.fields, opts.exclude)
    # print "constructed instance with password = %s" % self.instance.password

    exclude = self._get_validation_exclusions()

    # Foreign Keys being used to represent inline relationships
    # are excluded from basic field value validation. This is for two
    # reasons: firstly, the value may not be supplied (#12507; the
    # case of providing new values to the admin); secondly the
    # object being referred to may not yet fully exist (#12749).
    # However, these fields *must* be included in uniqueness checks,
    # so this can't be part of _get_validation_exclusions().
    for f_name, field in self.fields.items():
        if isinstance(field, InlineForeignKeyField):
            exclude.append(f_name)

    from django.core.exceptions import ValidationError
    # Clean the model instance's fields.
    try:
        self.instance.full_clean(exclude)
    except ValidationError, e:
        self._update_errors(e.update_error_dict(None))

@patch(BaseForm, '_clean_form')
def clean_form_with_field_errors(original_function, self):
    """
    Allow BaseForm._clean_form to report errors on individual fields,
    instead of the whole form, like this:
    
    raise ValidationError({'password': 'Incorrect password'})
    
    The standard version only works on the whole form:
    https://code.djangoproject.com/ticket/16423
    """
    
    from django.core.exceptions import ValidationError
    try:
        self.cleaned_data = self.clean()
    except ValidationError, e:
        if hasattr(e, 'message_dict'):
            for field, error_strings in e.message_dict.items():
                self._errors[field] = self.error_class(error_strings)
        else:
            self._errors[NON_FIELD_ERRORS] = self.error_class(e.messages)

pp = PrettyPrinter()

@patch(RegexURLResolver, 'reverse')
def reverse_with_debugging(original_function, self, lookup_view, *args, **kwargs):
    """
    Show all the patterns in the reverse_dict if a reverse lookup fails,
    to help figure out why.
    """
    
    try:
        return original_function(self, lookup_view, *args, **kwargs)
    except NoReverseMatch as e:
    	# if the function is a callable, it might be a wrapper
    	# function which isn't identical (comparable) to another
    	# wrapping of the same function
    	# import pdb; pdb.set_trace()
    	
        if lookup_view in self.reverse_dict:
            raise NoReverseMatch(str(e) + (" Possible match: %s" %
                (self.reverse_dict[lookup_view],)))
        else:
            if callable(lookup_view):
                raise NoReverseMatch(str(e) + "\n" +
                    ("No such key %s in %s\n\n" % (lookup_view,
                        [k for k in self.reverse_dict.keys() if callable(k)])) +
                    ("Complete reverse map: %s\n" % pp.pformat(self.reverse_dict)))
            else:
                raise NoReverseMatch(str(e) + "\n" +
                	("No such key %s in %s\n" % (lookup_view,
                        [k for k in self.reverse_dict.keys() if not callable(k)])) +
                	("Complete reverse map: %s\n" % pp.pformat(self.reverse_dict)))
if '_reverse_with_prefix' in dir(RegexURLResolver):
    # support for Django 1.4:
    patch(RegexURLResolver, '_reverse_with_prefix', reverse_with_debugging)

@after(RegexURLResolver, '_populate')
def populate_reverse_dict_with_module_function_names(self):
	from django.utils.translation import get_language
	language_code = get_language()
	reverse_dict = self._reverse_dict[language_code]
	for pattern in reversed(self.url_patterns):
		if not isinstance(pattern, RegexURLResolver):
			# import pdb; pdb.set_trace()
			for reverse_item in reverse_dict.getlist(pattern.callback):
				function_name = "%s.%s" % (pattern.callback.__module__,
					pattern.callback.__name__)
				reverse_dict.appendlist(function_name, reverse_item)

class FieldlineWithCustomReadOnlyField(object):
    """
    Custom replacement for Fieldline that allows fields in the Admin
    interface to render their own read-only view if they like.
    """
    
    def __init__(self, form, field, readonly_fields=None, model_admin=None):
        self.form = form # A django.forms.Form instance
        if not hasattr(field, "__iter__"):
            self.fields = [field]
        else:
            self.fields = field
        self.model_admin = model_admin
        if readonly_fields is None:
            readonly_fields = ()
        self.readonly_fields = readonly_fields

    def __iter__(self):
        for i, field in enumerate(self.fields):
            if field in self.readonly_fields:
                from admin import CustomAdminReadOnlyField
                yield CustomAdminReadOnlyField(self.form, field, is_first=(i == 0),
                    model_admin=self.model_admin)
            else:
                yield AdminField(self.form, field, is_first=(i == 0))

    def errors(self):
        return mark_safe(u'\n'.join([self.form[f].errors.as_ul() for f in self.fields if f not in self.readonly_fields]).strip('\n'))
django.contrib.admin.helpers.Fieldline = FieldlineWithCustomReadOnlyField

from django.db.backends.creation import BaseDatabaseCreation
# @patch(BaseDatabaseCreation, 'destroy_test_db')
def destroy_test_db_disabled(original_function, self, test_database_name,
    verbosity):
    """
    Temporarily disable the deletion of a test database, for post-mortem
    examination.
    """
    test_database_name = self.connection.settings_dict['NAME']
    if verbosity >= 1:
        print("Not destroying test database for alias '%s' (%s)..." % (
            self.connection.alias, test_database_name))

if not hasattr(auth_models.Group, 'natural_key'):
    """
    Allow group lookups by name in fixtures, until
    https://code.djangoproject.com/ticket/13914 lands.
    """
    
    from django.db import models as db_models
    class GroupManagerWithNaturalKey(db_models.Manager):
        def get_by_natural_key(self, name):
            return self.get(name=name)
    # print "auth_models.Group.objects = %s" % auth_models.Group.objects
    del auth_models.Group._default_manager
    GroupManagerWithNaturalKey().contribute_to_class(auth_models.Group, 'objects')
    def group_natural_key(self):
        return (self.name,)
    auth_models.Group.natural_key = group_natural_key

def Deserializer_with_debugging(original_function, object_list, **options):
    from django.core.serializers.python import _get_model
    from django.db import DEFAULT_DB_ALIAS
    from django.utils.encoding import smart_unicode
    from django.conf import settings

    print "loading all: %s" % object_list

    db = options.pop('using', DEFAULT_DB_ALIAS)
    db_models.get_apps()
    for d in object_list:
        print "loading %s" % d
        
        # Look up the model and starting build a dict of data for it.
        Model = _get_model(d["model"])
        data = {Model._meta.pk.attname : Model._meta.pk.to_python(d["pk"])}
        m2m_data = {}

        # Handle each field
        for (field_name, field_value) in d["fields"].iteritems():
            if isinstance(field_value, str):
                field_value = smart_unicode(field_value, options.get("encoding", settings.DEFAULT_CHARSET), strings_only=True)

            field = Model._meta.get_field(field_name)

            # Handle M2M relations
            if field.rel and isinstance(field.rel, db_models.ManyToManyRel):
                print "  field = %s" % field
                print "  field.rel = %s" % field.rel
                print "  field.rel.to = %s" % field.rel.to
                print "  field.rel.to._default_manager = %s" % (
                    field.rel.to._default_manager)
                print "  field.rel.to.objects = %s" % (
                    field.rel.to.objects)

                if hasattr(field.rel.to._default_manager, 'get_by_natural_key'):
                    def m2m_convert(value):
                        if hasattr(value, '__iter__'):
                            return field.rel.to._default_manager.db_manager(db).get_by_natural_key(*value).pk
                        else:
                            return smart_unicode(field.rel.to._meta.pk.to_python(value))
                else:
                    m2m_convert = lambda v: smart_unicode(field.rel.to._meta.pk.to_python(v))
                m2m_data[field.name] = [m2m_convert(pk) for pk in field_value]
                for i, pk in enumerate(field_value):
                    print "  %s: converted %s to %s" % (field.name,
                        pk, m2m_data[field.name][i])
    
    result = original_function(object_list, **options)
    print "  result = %s" % result
    import traceback
    traceback.print_stack()
    return result
# patch(django.core.serializers.python, 'Deserializer',
#     Deserializer_with_debugging)

def save_with_debugging(original_function, self, save_m2m=True, using=None):
    print "%s.save(save_m2m=%s, using=%s)" % (self, save_m2m, using)
    original_function(self, save_m2m, using)
# patch(django.core.serializers.base.DeserializedObject, 'save',
#     save_with_debugging)

def ContextList_keys(self):
    keys = set()
    for subcontext in self:
        for dict in subcontext:
            keys |= set(dict.keys())
    return keys
ContextList.keys = ContextList_keys

def configure_with_debugging(original_function, self,
    default_settings=global_settings, **options):
    print "LazySettings configured: %s, %s" % (default_settings, options)
    import traceback
    traceback.print_stack()
    return original_function(self, default_settings, **options)
# patch(LazySettings, 'configure', configure_with_debugging)

def setup_with_debugging(original_function, self):
    print "LazySettings setup:"
    import traceback
    traceback.print_stack()
    return original_function(self)
# patch(LazySettings, '_setup', setup_with_debugging)

# before(ChangeList, 'get_results')(breakpoint)
# @before(ChangeList, 'get_results')
"""
def get_results_with_debugging(self, request):
    print "get_results query = %s" % object.__str__(self.query_set.query)
"""

# from django.forms.forms import BoundField
# before(BoundField, 'value')(breakpoint)

# Until a patch for 6707 lands: https://code.djangoproject.com/ticket/6707
"""
from django.db.models.fields.related import ReverseManyRelatedObjectsDescriptor
def related_objects_set_without_clear(original_function, self, instance,
    new_values):
    
    if instance is None:
        raise AttributeError("Manager must be accessed via instance")

    if not self.field.rel.through._meta.auto_created:
        opts = self.field.rel.through._meta
        raise AttributeError("Cannot set values on a ManyToManyField which specifies an intermediary model.  Use %s.%s's Manager instead." % (opts.app_label, opts.object_name))

    manager = self.__get__(instance)
    old_values = manager.all()
    values_to_remove = [v for v in old_values 
        if v not in new_values]
    manager.remove(*values_to_remove)
patch(ReverseManyRelatedObjectsDescriptor, '__set__',
    related_objects_set_without_clear)
"""

def AutoField_to_python_with_improved_debugging(original_function, self, value):
    try:
        return original_function(self, value)
    except (TypeError, ValueError):
        from django.core.exceptions import ValidationError
        raise ValidationError(self.error_messages['invalid'] +
            ": %s.%s is not allowed to have value '%s'" % 
            (self.model, self.name, value))
# print "before patch: IntranetUser.id.to_python = %s" % IntranetUser.id.to_python
patch(AutoField, 'to_python', AutoField_to_python_with_improved_debugging)
# print "after patch: IntranetUser.id.to_python = %s" % IntranetUser.id.to_python

# Show the filename that contained the template error
"""
@patch(django.template.loader, 'render_to_string')
def template_loader_render_to_string_with_debugging(original_function,
    template_name, dictionary=None, context_instance=None):
    
    try:
        return original_function(template_name, dictionary, context_instance)
    except Exception as e:
        import sys
        raise Exception, "Failed to render template: %s: %s" % \
            (template_name, e), sys.exc_info()[2]
"""

# Show the filename that contained the template error
@patch(django.template.base.Template, 'render')
def template_render_with_debugging(original_function, self, context):
    try:
        return original_function(self, context)
    except Exception as e:
        import sys
        raise Exception, "Failed to render template: %s: %s" % \
            (self.name, e), sys.exc_info()[2]

@patch(django.template.defaulttags.URLNode, 'render')
def urlnode_render_with_debugging(original_function, self, context):
    if not self.view_name.resolve(context):
        raise Exception(("Failed to resolve %s in context: did you " +
            "forget to enclose view name in quotes? Context is: %s") %
            (self.view_name, context))
        
    try:
        return original_function(self, context)
    except NoReverseMatch as e:
        raise Exception(("Failed to reverse %s in context %s (did you " +
            "forget to enclose view name in quotes?): the exception was: %s") %
            (self.view_name, context, e))

from django.db.models.fields import DateTimeField
@before(DateTimeField, 'get_prep_value')
def DateTimeField_get_prep_value_check_for_naive_datetime(self, value):
    value = self.to_python(value)
    from django.conf import settings
    from django.utils import timezone
    if value is not None and settings.USE_TZ and timezone.is_naive(value):
        raise ValueError(("DateTimeField %s.%s received a " +
            "naive datetime (%s) while time zone support is " +
            "active.") % (self.model.__name__, self.name, value))

from django.template.base import Variable
@patch(Variable, '__init__')
def Variable_init_with_underscores_allowed(original_function, self, var):
    from django.conf import settings
    # for security reasons, production deployments are not allowed to
    # render variable names containing underscores anyway.
    if not settings.DEBUG:
        return original_function(self, var)
    
    self.var = var
    self.literal = None
    self.lookups = None
    self.translate = False
    self.message_context = None

    try:
        # First try to treat this variable as a number.
        #
        # Note that this could cause an OverflowError here that we're not
        # catching. Since this should only happen at compile time, that's
        # probably OK.
        self.literal = float(var)

        # So it's a float... is it an int? If the original value contained a
        # dot or an "e" then it was a float, not an int.
        if '.' not in var and 'e' not in var.lower():
            self.literal = int(self.literal)

        # "2." is invalid
        if var.endswith('.'):
            raise ValueError

    except ValueError:
        # A ValueError means that the variable isn't a number.
        if var.startswith('_(') and var.endswith(')'):
            # The result of the lookup should be translated at rendering
            # time.
            self.translate = True
            var = var[2:-1]
        # If it's wrapped with quotes (single or double), then
        # we're also dealing with a literal.
        try:
            from django.utils.text import unescape_string_literal
            self.literal = mark_safe(unescape_string_literal(var))
        except ValueError:
            # Otherwise we'll set self.lookups so that resolve() knows we're
            # dealing with a bonafide variable
            """
            if var.find(VARIABLE_ATTRIBUTE_SEPARATOR + '_') > -1 or var[0] == '_':
                raise TemplateSyntaxError("Variables and attributes may "
                                          "not begin with underscores: '%s'" %
                                          var)
            """
            from django.template.base import VARIABLE_ATTRIBUTE_SEPARATOR
            self.lookups = tuple(var.split(VARIABLE_ATTRIBUTE_SEPARATOR))

# temporary patch for https://code.djangoproject.com/ticket/16955
from django.db.models.sql.query import Query
@before(Query, 'add_filter')
def add_filter_add_value_capture(self, filter_expr, *args, **kwargs):
    arg, value = filter_expr
    self._captured_value_for_monkeypatch = value
@after(Query, 'add_filter')
def add_filter_remove_value_capture(self, value, *args, **kwargs):
    delattr(self, '_captured_value_for_monkeypatch')
@patch(Query, 'setup_joins')
def setup_joins_with_value_type_check(original_function, self, *args, **kwargs):
    results = original_function(self, *args, **kwargs)
    value = getattr(self, '_captured_value_for_monkeypatch', None)
    from users.models import Price
    # if results[0].model == Price:
    #     import pdb; pdb.set_trace()
    if value:
        field = results[0]
        target = results[1]
        from django.db.models.fields.related import RelatedField
        from django.db.models import Model
        if (isinstance(field, RelatedField) and isinstance(value, Model) and 
            not isinstance(value, target.model)): 
            raise TypeError, "'%s' instance expected" % target.model._meta.object_name
    return results         

from django.contrib.auth.forms import ReadOnlyPasswordHashField
@patch(ReadOnlyPasswordHashField, 'bound_data')
def bound_data_with_bug_19611_patch(original_function, self, data, initial):
    """
    This widget has no fields, so data will always be None, so return
    the initial value always.
    """
    return initial

# patch for deprecated https://code.djangoproject.com/ticket/20030,
# to get us through django 1.5
from django.db.backends import BaseDatabaseWrapper, BaseDatabaseFeatures
@patch(BaseDatabaseFeatures, 'supports_transactions')
def supports_transactions_with_bug_20030_patch(original_function, self):
    self.connection._in_supports_transactions = True
    try:
        return original_function(self)
    finally:
        self.connection._in_supports_transactions = False
@patch(BaseDatabaseWrapper, 'leave_transaction_management')
def leave_transaction_management_with_bug_20030_patch(original_function, self):
    if getattr(self, '_in_supports_transactions', False) and self._dirty:
        # supports_transactions not finished, must have been aborted
        # by an exception, so swallow any exceptions thrown by
        # leave_transaction_management()
        try:
            original_function(self)
        except:
            # ignore exceptions
            pass
    else:
        original_function(self)

# we can't descend a relationship into a different database, no matter
# how much we might want to.
"""
from django.db.models.sql.compiler import SQLCompiler
@patch(SQLCompiler, 'fill_related_selections')
def fill_related_selections_without_different_dbs(
    original_fill_related_selections, self, opts=None, root_alias=None,
    cur_depth=1, used=None, requested=None, restricted=None, nullable=None,
    dupe_set=None, avoid_set=None):
    
    if not opts:
        opts = self.query.get_meta()
        root_alias = self.query.get_initial_alias()
        self.query.related_select_cols = []
        self.query.related_select_fields = []
        
    compiler = self
    
    # In the scope of executing fill_related_selections() only, we patch the
    # opts argument's get_fields_with_model() method so that certain fields
    # are not returned: the ones which are foreign keys to different
    # databases.
    
    def get_fields_with_model_without_different_dbs(
        original_get_fields_with_model):
        
        from django.db.models.fields.related import ForeignKey
        return [(field, model)
            for field, model in original_get_fields_with_model(opts)
            if not isinstance(field, ForeignKey)
            or field.rel.to._meta.db_tablespace != compiler.using]

    # import pdb; pdb.set_trace()

    with patch(opts, 'get_fields_with_model',
        get_fields_with_model_without_different_dbs):
        
        return original_fill_related_selections(self, opts, root_alias,
            cur_depth, used, requested, restricted, nullable, dupe_set,
            avoid_set)
"""
        
"""
from django.db.models import query_utils
@patch(query_utils, 'select_related_descend')
def select_related_descend_without_different_dbs(original_function,
    field, restricted, requested, load_fields, reverse=False):
    
    # we should really use the actual connection ('using') for both
    # sides, but how do we access connections from here?
    from django.db import connections, DEFAULT_DB_ALIAS
    
    if field.model.
')
"""

# Add a patch to ModelAdmin to allow getting a list of related objects
# to be deleted.
from django.contrib.admin.options import ModelAdmin
@insert(ModelAdmin, 'get_deleted_objects')
def ModelAdmin_get_deleted_objects(original_function, self, objs, opts,
    request, using):
    
    """
    Find all objects related to ``objs`` that should also be deleted. ``objs``
    must be a homogenous iterable of objects (e.g. a QuerySet).

    Returns a nested list of strings suitable for display in the
    template with the ``unordered_list`` filter.

    By default this just passes the request on to 
    django.contrib.admin.util.get_deleted_objects, but this method exists
    to allow subclasses to override the permissions required to delete
    things.
    """
    from django.contrib.admin.util import get_deleted_objects
    return get_deleted_objects(objs, opts, request.user, self.admin_site, using)

# Patch ModelAdmin.delete_view() to call the instance method 
# get_deleted_objects() instead of the global function, to allow overriding it.
from django.contrib.admin.options import csrf_protect_m
from django.db import transaction
@patch(ModelAdmin, 'delete_view')
@csrf_protect_m
@transaction.commit_on_success
def ModelAdmin_delete_view_with_self_get_deleted_objects(self, request,
    object_id, extra_context=None):
    
    "The 'delete' admin view for this model."
    opts = self.model._meta
    app_label = opts.app_label
    
    from django.contrib.admin.util import unquote # , flatten_fieldsets, get_deleted_objects, model_format_dict
    obj = self.get_object(request, unquote(object_id))

    if not self.has_delete_permission(request, obj):
        from django.core.exceptions import PermissionDenied
        raise PermissionDenied

    from django.utils.encoding import force_unicode
    from django.utils.html import escape
    if obj is None:
        from django.http import Http404 
        raise Http404(_('%(name)s object with primary key %(key)r does not exist.') % {'name': force_unicode(opts.verbose_name), 'key': escape(object_id)})

    from django.db import router
    using = router.db_for_write(self.model)

    # Populate deleted_objects, a data structure of all related objects that
    # will also be deleted.
    (deleted_objects, perms_needed, protected) = self.get_deleted_objects(
        [obj], opts, request, using)

    if request.POST: # The user has already confirmed the deletion.
        if perms_needed:
            raise PermissionDenied
        
        obj_display = force_unicode(obj)
        self.log_deletion(request, obj, obj_display)
        self.delete_model(request, obj)

        self.message_user(request, _('The %(name)s "%(obj)s" was deleted successfully.') % {'name': force_unicode(opts.verbose_name), 'obj': force_unicode(obj_display)})

        from django.http import HttpResponseRedirect
        from django.core.urlresolvers import reverse
        if not self.has_change_permission(request, None):
            return HttpResponseRedirect(reverse('admin:index',
                                                current_app=self.admin_site.name))
        return HttpResponseRedirect(reverse('admin:%s_%s_changelist' %
                                    (opts.app_label, opts.module_name),
                                    current_app=self.admin_site.name))

    object_name = force_unicode(opts.verbose_name)

    if perms_needed or protected:
        title = _("Cannot delete %(name)s") % {"name": object_name}
    else:
        title = _("Are you sure?")

    context = {
        "title": title,
        "object_name": object_name,
        "object": obj,
        "deleted_objects": deleted_objects,
        "perms_lacking": perms_needed,
        "protected": protected,
        "opts": opts,
        "app_label": app_label,
    }
    context.update(extra_context or {})

    return TemplateResponse(request, self.delete_confirmation_template or [
        "admin/%s/%s/delete_confirmation.html" % (app_label, opts.object_name.lower()),
        "admin/%s/delete_confirmation.html" % app_label,
        "admin/delete_confirmation.html"
    ], context, current_app=self.admin_site.name)
