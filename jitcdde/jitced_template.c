# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-pedantic"
# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# include <Python.h>
# include <numpy/arrayobject.h>
# pragma GCC diagnostic pop
#include <structmember.h>

# include <math.h>
# include <assert.h>
# include <stdbool.h>
# include <stdio.h>

# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"

# define TYPE_INDEX NPY_DOUBLE

unsigned int const dimension={{n}};

typedef struct anchor
{
	double time;
	double state[{{n}}];
	double diff[{{n}}];
	struct anchor * next;
	struct anchor * previous;
} anchor;

typedef struct
{
	PyObject_HEAD
	anchor * current;
	anchor * first_anchor;
	anchor * last_anchor;
	anchor ** anchor_mem;
	anchor ** current_anchor;
	double past_within_step;
	double old_new_y[{{n}}];
	double error[{{n}}];
} dde_integrator;

void append_anchor(
	dde_integrator * const self,
	double const time,
	double state[{{n}}],
	double diff[{{n}}])
{
	anchor * new_anchor = malloc(sizeof(anchor));
	assert(new_anchor!=NULL);
	new_anchor->time = time;
	memcpy(new_anchor->state, state, {{n}}*sizeof(double));
	memcpy(new_anchor->diff, diff, {{n}}*sizeof(double));
	new_anchor->next = NULL;
	new_anchor->previous = self->last_anchor;
	
	if (self->last_anchor)
	{
		assert(self->last_anchor->next==NULL);
		self->last_anchor->next = new_anchor;
	}
	else
	{
		assert(self->first_anchor==NULL);
		self->first_anchor = new_anchor;
	}
	
	self->last_anchor = new_anchor;
}

void remove_first_anchor(dde_integrator * const self)
{
	anchor * old_first_anchor = self->first_anchor;
	
	self->first_anchor = old_first_anchor->next;
	if (self->first_anchor)
		self->first_anchor->previous = NULL;
	else
		self->last_anchor = NULL;
	
	free(old_first_anchor);
}

void replace_last_anchor(
	dde_integrator * const self,
	double const time,
	double state[{{n}}],
	double diff[{{n}}]
)
{
	memcpy(self->old_new_y , self->last_anchor->state, {{n}}*sizeof(double));
	self->last_anchor->time = time;
	memcpy(self->last_anchor->state, state, {{n}}*sizeof(double));
	memcpy(self->last_anchor->diff, diff, {{n}}*sizeof(double));
}

static PyObject * get_t(dde_integrator const * const self)
{
	return PyFloat_FromDouble(self->current->time);
}

anchor get_past_anchors(dde_integrator * const self, double const t)
{
	anchor * ca = *(self->current_anchor);
	
	while ( (ca->time > t) && (ca->previous) )
		ca = ca->previous;
	
	assert(ca->next != NULL);
	
	while ( (ca->next->time < t) && (ca->next->next) )
		ca = ca->next;
	
	if (t > self->current->time)
		self->past_within_step = fmax(self->past_within_step,t-self->current->time);
	
	*(self->current_anchor) = ca;
	self->current_anchor++;
	return *ca;
}

double get_past_value(
	dde_integrator const * const self,
	double const t,
	int unsigned const index,
	anchor const v)
{
	anchor w = *(v.next);
	double q = w.time-v.time;
	double x = (t - v.time)/q;
	double a = v.state[index];
	double b = v.diff[index] * q;
	double c = w.state[index];
	double d = w.diff[index] * q;
	
	return (1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x*x) + c;
}

static PyObject * get_recent_state(dde_integrator const * const self, PyObject * args)
{
	assert(self->last_anchor);
	assert(self->first_anchor);

	double t;
	if (!PyArg_ParseTuple(args, "d", &t))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	npy_intp dims[1] = {dimension};
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	PyArrayObject * result = (PyArrayObject *)PyArray_SimpleNew(1, dims, TYPE_INDEX);
	# pragma GCC diagnostic pop
	
	anchor w = *(self->last_anchor);
	anchor v = *(w.previous);
	double q = w.time-v.time;
	double x = (t - v.time)/q;
	
	for (int index=0; index<{{n}}; index++)
	{
		double a = v.state[index];
		double b = v.diff[index] * q;
		double c = w.state[index];
		double d = w.diff[index] * q;
	
		* (double *) PyArray_GETPTR1(result, index) = 
				(1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x*x) + c;
	}
	
	# pragma GCC diagnostic push
	# pragma GCC diagnostic ignored "-pedantic"
	return (PyObject *) result;
	# pragma GCC diagnostic pop
}


# define set_dy(i, value) (dY[i] = value)
# define current_y(i) (y[i])
# define past_y(t, i, anchor) (get_past_value(self, t, i, anchor))
# define anchors(t) (get_past_anchors(self, t))


# include "f_definitions.c"
static PyObject * eval_f(
	dde_integrator * const self,
	double const t,
	double y[{{n}}],
	double dY[{{n}}])
{
	self->current_anchor = self->anchor_mem;
	# include "f.c"
// 	set_dy(0, -0.1*current_y(0) + 0.25*past_y(t - 15, 0, anchors(t - 15))/(pow(past_y(t - 15, 0, anchors(t - 15)), 10) + 1.0));
	Py_RETURN_NONE;
}

static PyObject * get_next_step(dde_integrator * const self, PyObject * args)
{
	assert(self->last_anchor);
	assert(self->first_anchor);

	double delta_t;
	if (!PyArg_ParseTuple(args, "d", &delta_t))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	self->past_within_step = 0.0;
	# define k_1 self->current->diff
	double argument[{{n}}];
	
	double k_2[{{n}}];
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->current->state[i] + 0.5*delta_t*k_1[i];
	eval_f(self, self->current->time+0.5*delta_t, argument, k_2);
	
	double k_3[{{n}}];
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->current->state[i] + 0.75*delta_t*k_2[i];
	eval_f(self, self->current->time+0.75*delta_t, argument, k_3);
	
	double new_y[{{n}}];
	for (int i=0; i<{{n}}; i++)
		new_y[i] = self->current->state[i] + (delta_t/9.) * (2*k_1[i]+3*k_2[i]+4*k_3[i]);
	
	double new_t = self->current->time + delta_t;
	
	double k_4[{{n}}];
	# define new_diff k_4
	eval_f(self, new_t, new_y, new_diff);
	
	for (int i=0; i<{{n}}; i++)
		self->error[i] = (5*k_1[i]-6*k_2[i]-8*k_3[i]+9*k_4[i]) * (1/72.);
	
	if (self->last_anchor == self->current)
		append_anchor(self, new_t, new_y, new_diff);
	else
		replace_last_anchor(self, new_t, new_y, new_diff);
	
	assert(self->first_anchor);
	assert(self->last_anchor);
	Py_RETURN_NONE;
}

static PyObject * get_p(dde_integrator const * const self, PyObject * args)
{
	double atol;
	double rtol;
	if (!PyArg_ParseTuple(args, "dd", &atol, &rtol))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	double p=0.0;
	for (int i=0; i<{{n}}; i++)
	{
		double x = fabs(self->error[i]/(atol+rtol*fabs(self->last_anchor->state[i])));
		if (x>p)
			p = x;
	}
	
	return PyFloat_FromDouble(p);
}

// result = np.max(np.abs(self.error)/(atol + rtol*np.abs(self.past[-1][1])))

static PyObject * check_new_y_diff(dde_integrator const * const self, PyObject * args)
{
	double atol;
	double rtol;
	if (!PyArg_ParseTuple(args, "dd", &atol, &rtol))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	bool result = true;
	for (int i=0; i<{{n}}; i++)
	{
		double difference = fabs(self->last_anchor->state[i] - self->old_new_y[i]);
		double tolerance = atol + fabs(rtol*self->last_anchor->state[i]);
		result &= (tolerance >= difference);
	}
	
	return PyBool_FromLong(result);
}

static PyObject * accept_step(dde_integrator * const self)
{
	self->current = self->last_anchor;
	Py_RETURN_NONE;
}

static PyObject * forget(dde_integrator * const self, PyObject * args)
{
	double delay;
	if (!PyArg_ParseTuple(args, "d", &delay))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	double threshold = self->current->time - delay;
	assert(self->first_anchor != self->last_anchor);
	while (self->first_anchor->next->time < threshold)
		remove_first_anchor(self);
	
	Py_RETURN_NONE;
}

static void dde_integrator_dealloc(dde_integrator * const self)
{
	while (self->first_anchor)
		remove_first_anchor(self);
	free(self->anchor_mem);
	
	{% if Python_version==3: %}
	Py_TYPE(self)->tp_free((PyObject *)self);
	{% elif Python_version==2: %}
	self->ob_type->tp_free((PyObject*)self);
	{% endif %}
}

static int initiate_past_from_list(dde_integrator * const self, PyObject * const past)
{
	for (Py_ssize_t i=0; i<PyList_Size(past); i++)
	{
		PyObject * pyanchor = PyList_GetItem(past,i);
		double time;
		PyArrayObject * pystate;
		PyArrayObject * pydiff;
		if (!PyArg_ParseTuple(pyanchor, "dO!O!", &time, &PyArray_Type, &pystate, &PyArray_Type, &pydiff))
		{
			PyErr_SetString(PyExc_ValueError,"Wrong input.");
			return 0;
		}
		
		double state[{{n}}];
		for (int i=0; i<{{n}}; i++)
			state[i] = * (double *) PyArray_GETPTR1(pystate,i);
		
		double diff[{{n}}];
		for (int i=0; i<{{n}}; i++)
			diff[i] = * (double *) PyArray_GETPTR1(pydiff,i);
		
		append_anchor(self, time, state, diff);
	}
	
	return 1;
}

static int dde_integrator_init(dde_integrator * self, PyObject * args)
{
	unsigned int anchor_mem_length;
	PyObject * past;
	if (!PyArg_ParseTuple(args, "O!I", &PyList_Type, &past, &anchor_mem_length))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return -1;
	}
	
	self->first_anchor = NULL;
	self->last_anchor = NULL;
	
	if (!initiate_past_from_list(self, past))
		return -1;
	self->current = self->last_anchor;
	assert(self->first_anchor != self->last_anchor);
	assert(self->first_anchor != NULL);
	assert(self->last_anchor != NULL);
	
	self->anchor_mem = malloc(anchor_mem_length*sizeof(anchor *));
	assert(self->anchor_mem != NULL);
	for (int i=0; i<anchor_mem_length; i++)
		self->anchor_mem[i] = self->last_anchor->previous;
	
	return 0;
}

static PyMemberDef dde_integrator_members[] = {
 	{"past_within_step", T_DOUBLE, offsetof(dde_integrator, past_within_step), 0, "past_within_step"},
	{NULL}  /* Sentinel */
};

static PyMethodDef dde_integrator_methods[] = {
	{"get_t", (PyCFunction) get_t, METH_NOARGS, NULL},
	{"get_recent_state", (PyCFunction) get_recent_state, METH_VARARGS, NULL},
	{"get_next_step", (PyCFunction) get_next_step, METH_VARARGS, NULL},
	{"get_p", (PyCFunction) get_p, METH_VARARGS, NULL},
	{"check_new_y_diff", (PyCFunction) check_new_y_diff, METH_VARARGS, NULL},
	{"accept_step", (PyCFunction) accept_step, METH_NOARGS, NULL},
	{"forget", (PyCFunction) forget, METH_VARARGS, NULL},
	{NULL, NULL, 0, NULL}
};


static PyTypeObject dde_integrator_type = {
	{% if Python_version==3: %}
	PyVarObject_HEAD_INIT(NULL, 0)
	{% elif Python_version==2: %}
	PyObject_HEAD_INIT(NULL)
	0, 
	{% endif %}
	"_jitced.dde_integrator",
	sizeof(dde_integrator), 
	0,                         // tp_itemsize 
	(destructor) dde_integrator_dealloc,
	0,                         // tp_print 
	0,0,0,0,0,0,0,0,0,0,0,0,   // ... 
	0,                         // tp_as_buffer 
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	0,                         // tp_doc 
	0,0,0,0,0,                 // ... 
	0,                         // tp_iternext 
	dde_integrator_methods,
	dde_integrator_members,
	0,                         // tp_getset 
	0,0,0,0,                   // ...
	0,                         // tp_dictoffset 
	(initproc) dde_integrator_init,
	0,                         // tp_alloc 
	PyType_GenericNew,
};

static PyMethodDef {{module_name}}_methods[] = {
{NULL, NULL, 0, NULL}
};


{% if Python_version==3: %}

static struct PyModuleDef moduledef =
{
        PyModuleDef_HEAD_INIT,
        "{{module_name}}",
        NULL,
        -1,
        {{module_name}}_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_{{module_name}}(void)
{
	dde_integrator_type.tp_new = PyType_GenericNew;
	if (PyType_Ready(&dde_integrator_type) < 0)
		return NULL;
	
	PyObject * module = PyModule_Create(&moduledef);
	
	Py_INCREF(&dde_integrator_type);
	PyModule_AddObject(module, "dde_integrator", (PyObject *)&dde_integrator_type);
	
	import_array();
	
	return module;
}

{% elif Python_version==2: %}

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC init{{module_name}}(void)
{
	if (PyType_Ready(&dde_integrator_type) < 0)
		return;
	
	PyObject * module = Py_InitModule("{{module_name}}", {{module_name}}_methods);
	
	if (module == NULL)
		return;
	
	Py_INCREF(&dde_integrator_type);
	
	PyModule_AddObject(module, "dde_integrator", (PyObject*) &dde_integrator_type);
	
	import_array();
}

{% endif %}
