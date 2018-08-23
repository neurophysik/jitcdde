# define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
# include <Python.h>
# include <numpy/arrayobject.h>
# include <math.h>
# include <structmember.h>
# include <assert.h>
# include <stdbool.h>

# define TYPE_INDEX NPY_DOUBLE

# define NORM_THRESHOLD (1e-30)

static inline void * safe_malloc(size_t size)
{
	void * pointer = malloc(size);
	if (pointer == NULL)
		PyErr_SetString(PyExc_MemoryError,"Could not allocate memory.");
	return pointer;
}

typedef struct anchor
{
	double time;
	double state[{{n}}];
	double diff[{{n}}];
	struct anchor * next;
	struct anchor * previous;
	{% if (n_basic != n) or tangent_indices: %}
	double sp_matrix[4][4];
	{% endif %}
} anchor;

typedef struct
{
	PyObject_HEAD
	anchor * current;
	anchor * first_anchor;
	anchor * last_anchor;
	{% if anchor_mem_length: %}
	anchor ** anchor_mem;
	anchor ** anchor_mem_cursor;
	{% endif %}
	double past_within_step;
	anchor * old_last;
	double error[{{n}}];
	double last_actual_step_start;
	{% for control_par in control_pars %}
	double parameter_{{control_par}};
	{% endfor %}
} dde_integrator;

void append_anchor(dde_integrator * const self, anchor * const new_anchor)
{
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

void replace_last_anchor(dde_integrator * const self, anchor * const new_anchor)
{
	free(self->old_last);
	self->old_last = self->last_anchor;
	new_anchor->previous = self->old_last->previous;
	new_anchor->previous->next = new_anchor;
	self->last_anchor = new_anchor;
	new_anchor->next = NULL;
}

{% if control_pars|length %}

static PyObject * set_parameters(dde_integrator * const self, PyObject * args)
{
	if (!PyArg_ParseTuple(
		args,
		"{{'d'*control_pars|length}}"
		{% for control_par in control_pars %}
		, &(self->parameter_{{control_par}})
		{% endfor %}
		))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	Py_RETURN_NONE;
}

{% endif %}

static PyObject * get_t(dde_integrator const * const self)
{
	return PyFloat_FromDouble(self->current->time);
}

{% if anchor_mem_length: %}
anchor get_past_anchors(dde_integrator * const self, double const t)
{
	assert (self->anchor_mem_cursor <= &( self->anchor_mem[{{anchor_mem_length}}-1] ));
	anchor ** this_cursor;
	
	#pragma omp atomic capture
	this_cursor = self->anchor_mem_cursor++;
	// Note that the above only ensures that no two threads operate on the same cursor and that there are no race conditions. If two calls of get_past_anchors are executed in the "wrong" order, they will get the "wrong" cursor, i.e., they probably have to search considerably longer to find the right anchors. As this does not affect the correctness of the results but only the runtime, it's okay to do this. It may void the speed boost from parallelising though. Hope is that even with parallelising there there is a stable order in which get_past_anchors is called and thus every call of get_past_anchor gets its unique cursor.
	
	anchor * ca = *this_cursor;
	while ( (ca->time > t) && (ca->previous) )
		ca = ca->previous;
	
	assert(ca->next != NULL);
	
	while ( (ca->next->time < t) && (ca->next->next) )
		ca = ca->next;
	
	if (t > self->current->time)
		#pragma omp critical(pws)
		self->past_within_step = fmax(self->past_within_step,t-self->current->time);
	
	*this_cursor = ca;
	return *ca;
}
{% endif %}


double get_past_value(
	dde_integrator const * const self,
	double const t,
	unsigned int const index,
	anchor const v)
{
	anchor const w = *(v.next);
	double const q = w.time-v.time;
	double const x = (t - v.time)/q;
	double const a = v.state[index];
	double const b = v.diff[index] * q;
	double const c = w.state[index];
	double const d = w.diff[index] * q;
	
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
	
	npy_intp dims[1] = { {{n}} };
	PyArrayObject * result = (PyArrayObject *)PyArray_SimpleNew(1, dims, TYPE_INDEX);
	
	anchor const w = *(self->last_anchor);
	anchor const v = *(w.previous);
	double const q = w.time-v.time;
	double const x = (t - v.time)/q;
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int index=0; index<{{n}}; index++)
	{
		double const a = v.state[index];
		double const b = v.diff[index] * q;
		double const c = w.state[index];
		double const d = w.diff[index] * q;
	
		* (double *) PyArray_GETPTR1(result, index) = 
				(1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x*x) + c;
	}
	
	return (PyObject *) result;
}

static PyObject * get_current_state(dde_integrator const * const self)
{
	assert(self->last_anchor);
	
	npy_intp dims[1] = { {{n}} };
	PyArrayObject * result = (PyArrayObject *)PyArray_SimpleNew(1, dims, TYPE_INDEX);
	
	for (int index=0; index<{{n}}; index++)
		* (double *) PyArray_GETPTR1(result, index) = self->last_anchor->state[index];
	
	return (PyObject *) result;
}

static PyObject * get_full_state(dde_integrator const * const self)
{
	PyObject * py_past = PyList_New(0);
    npy_intp dim[1] = { {{n}} };
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		PyList_Append(
			py_past,
			PyTuple_Pack(
				3,
				PyFloat_FromDouble(ca->time),
				PyArray_SimpleNewFromData(1, dim, TYPE_INDEX, ca->state),
				PyArray_SimpleNewFromData(1, dim, TYPE_INDEX, ca->diff ) 
				)
			);
	return py_past;
}


# define set_dy(i, value) (dY[i] = value)
# define current_y(i) (y[i])
# define past_y(t, i, anchor) (get_past_value(self, t, i, anchor))
# define anchors(t) (get_past_anchors(self, t))

# define get_f_helper(i) ((f_helper[i]))
# define set_f_helper(i,value) (f_helper[i] = value)
# define get_f_anchor_helper(i) ((f_anchor_helper[i]))
# define set_f_anchor_helper(i,value) (f_anchor_helper[i] = value)


{% if has_any_helpers: %}
# include "helpers_definitions.c"
{% endif %}
# include "f_definitions.c"
void eval_f(
	dde_integrator * const self,
	double const t,
	double y[{{n}}],
	double dY[{{n}}])
{
	{% if anchor_mem_length: %}
		self->anchor_mem_cursor = self->anchor_mem;
	{% endif %}
	
	{% if number_of_helpers>0: %}
	double f_helper[{{number_of_helpers}}];
	{% endif %}
	{% if number_of_anchor_helpers>0: %}
	anchor f_anchor_helper[{{number_of_anchor_helpers}}];
	{% endif %}
	
	{% if has_any_helpers>0: %}
	# include "helpers.c"
	{% endif %}
	# include "f.c"
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
	
	self->last_actual_step_start = self->current->time;
	
	anchor * new = safe_malloc(sizeof(anchor));
	
	self->past_within_step = 0.0;
	# define k_1 self->current->diff
	double argument[{{n}}];
	
	double k_2[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->current->state[i] + 0.5*delta_t*k_1[i];
	eval_f(self, self->current->time+0.5*delta_t, argument, k_2);
	
	double k_3[{{n}}];
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		argument[i] = self->current->state[i] + 0.75*delta_t*k_2[i];
	eval_f(self, self->current->time+0.75*delta_t, argument, k_3);
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		new->state[i] = self->current->state[i] + (delta_t/9.) * (2*k_1[i]+3*k_2[i]+4*k_3[i]);
	
	new->time = self->current->time + delta_t;
	
	# define k_4 new->diff
	eval_f( self, new->time, new->state, new->diff );
	
	#pragma omp parallel for schedule(dynamic, {{chunk_size}})
	for (int i=0; i<{{n}}; i++)
		self->error[i] = (5*k_1[i]-6*k_2[i]-8*k_3[i]+9*k_4[i]) * (1/72.);
	
	if (self->last_anchor == self->current)
		append_anchor(self,new);
	else
		replace_last_anchor(self,new);
	
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
		double error = fabs(self->error[i]);
		double tolerance = atol+rtol*fabs(self->last_anchor->state[i]);
		if (error!=0.0 || tolerance!=0.0)
		{
			double x = error/tolerance;
			if (x>p)
				p = x;
		}
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
	if (self->old_last == NULL)
		result = false;
	else
		for (int i=0; i<{{n}}; i++)
		{
			double difference = fabs(self->last_anchor->state[i] - self->old_last->state[i]);
			double tolerance = atol + fabs(rtol*self->last_anchor->state[i]);
			result &= (tolerance >= difference);
		}
	
	return PyBool_FromLong(result);
}

static PyObject * accept_step(dde_integrator * const self)
{
	self->current = self->last_anchor;
	free(self->old_last);
	self->old_last = NULL;
	Py_RETURN_NONE;
}

static PyObject * adjust_diff(dde_integrator * const self, PyObject * args)
{
	double shift_ratio;
	if (!PyArg_ParseTuple(args, "d", &shift_ratio))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	assert(self->last_anchor);
	assert(self->first_anchor);
	
	anchor * new = safe_malloc(sizeof(anchor));
	
	new->time = self->current->time;
	memcpy( new->state, self->current->state, sizeof(double[{{n}}]) );
	eval_f( self, self->current->time, self->current->state, new->diff );
	double const gap = self->current->time-self->current->previous->time;
	self->current->time -= shift_ratio*gap;
	
	append_anchor(self,new);
	self->current = self->last_anchor;
	
	assert(self->current->time > self->current->previous->time);
	assert(self->last_anchor==new);
	assert(self->first_anchor);
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
	double threshold = fmin(
		self->current->time - delay,
		self->last_actual_step_start
		);
	assert(self->first_anchor != self->last_anchor);
	while (self->first_anchor->next->time < threshold)
	{
		{% if anchor_mem_length: %}
		# ifndef NDEBUG
		for (int i=0; i<{{anchor_mem_length}}; i++)
			assert(self->anchor_mem[i] != self->first_anchor);
		# endif
		{% endif %}
		remove_first_anchor(self);
	}
	
	assert(self->first_anchor != self->last_anchor);
	
	Py_RETURN_NONE;
}

static void dde_integrator_dealloc(dde_integrator * const self)
{
	while (self->first_anchor)
		remove_first_anchor(self);
	free(self->old_last);
	{% if anchor_mem_length: %}
	free(self->anchor_mem);
	{% endif %}
	
	Py_TYPE(self)->tp_free((PyObject *)self);
}

static int initiate_past_from_list(dde_integrator * const self, PyObject * const past)
{
	for (Py_ssize_t i=0; i<PyList_Size(past); i++)
	{
		anchor * new = safe_malloc(sizeof(anchor));
		
		PyObject * pyanchor = PyList_GetItem(past,i);
		PyArrayObject * pystate;
		PyArrayObject * pydiff;
		if (!PyArg_ParseTuple(pyanchor, "dO!O!", &new->time, &PyArray_Type, &pystate, &PyArray_Type, &pydiff))
		{
			PyErr_SetString(PyExc_ValueError,"Wrong input.");
			return 0;
		}
		
		if ( (PyArray_TYPE(pystate) != NPY_DOUBLE) || (PyArray_TYPE(pydiff) != NPY_DOUBLE) )
		{
			PyErr_SetString(PyExc_ValueError,"Anchors must be float arrays.");
			return 0;
		}
		
		for (int i=0; i<{{n}}; i++)
			new->state[i] = * (double *) PyArray_GETPTR1(pystate,i);
		
		for (int i=0; i<{{n}}; i++)
			new->diff[i] = * (double *) PyArray_GETPTR1(pydiff,i);
		
		append_anchor(self, new);
	}
	
	return 1;
}

static int dde_integrator_init(dde_integrator * self, PyObject * args)
{
	PyObject * past;
	if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &past))
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
	self->last_actual_step_start = self->first_anchor->time;
	self->old_last = NULL;
	
	{% if anchor_mem_length: %}
	self->anchor_mem = safe_malloc({{anchor_mem_length}}*sizeof(anchor *));
	assert(self->anchor_mem != NULL);
	for (int i=0; i<{{anchor_mem_length}}; i++)
		self->anchor_mem[i] = self->last_anchor->previous;
	{% endif %}
	
	return 0;
}

// Functions for both, normal and transversal, Lyapunov exponents
{% if (n_basic != n) or tangent_indices: %}

void calculate_sp_matrix(
	dde_integrator const * const self,
	anchor * v
)
{
	anchor const * const w = v->next;
	double const q = w->time - v->time;
	
	double cq = q/420.;
	double cqq = cq*q;
	double cqqq = cqq*q;
	memcpy(v->sp_matrix, (double[4][4]){
		{ 156*cq  ,22*cqq  , 54*cq  ,-13*cqq  },
		{  22*cqq , 4*cqqq , 13*cqq , -3*cqqq },
		{  54*cq  ,13*cqq  ,156*cq  ,-22*cqq  },
		{ -13*cqq ,-3*cqqq ,-22*cqq ,  4*cqqq }
	}, sizeof(double[4][4]));
}

void calculate_partial_sp_matrix(
	dde_integrator const * const self,
	anchor * v,
	double const threshold
)
{
	anchor const * const w = v->next;
	double const q = w->time - v->time;
	double const z = (threshold - w->time) / q;
	
	double cq = q/420.;
	double cqq = cq*q;
	double cqqq = cqq*q;
	
	double z3 = z*z*z;
	double z4 = z*z3;
	double z5 = z*z4;
	
	double const h_1 = (- 120*z*z - 350*z - 252) * z5 * cqq ;
	double const h_2 = (-  60*z*z - 140*z -  84) * z5 * cqqq;
	double const h_3 = (- 120*z*z - 420*z - 378) * z5 * cq  ;
	double const h_4 = (-  70*z*z - 168*z - 105) * z4 * cqqq;
	double const h_6 = (          - 105*z - 140) * z3 * cqq ;
	double const h_7 = (          - 210*z - 420) * z3 * cq  ;
	double const h_5 = (2*h_2 + 3*h_4)/q;
	double const h_8 = - h_5 + h_7*q - h_6 - 0.5*(z*q)*(z*q);
	
	memcpy(v->sp_matrix, (double[4][4]){
		{  2*h_3   , h_1    , h_7-2*h_3      , h_5                },
		{    h_1   , h_2    , h_6-h_1        , h_2+h_4            },
		{ h_7-2*h_3, h_6-h_1, 2*h_3-2*h_7-z*q, h_8                },
		{   h_5    , h_2+h_4, h_8            , h_2+(h_5+h_6-h_1)*q}
	}, sizeof(double[4][4]));
}

void calculate_sp_matrices(dde_integrator const * const self, double const delay)
{
	double const threshold = self->current->time - delay;
	
	#pragma omp parallel
	#pragma omp single
	{
	anchor * ca = self->first_anchor;
	for (; ca->next->time<threshold; ca=ca->next)
		#pragma omp task firstprivate(ca)
		for (int i=0; i<4; i++)
			for (int j=0; j<4; j++)
				ca->sp_matrix[i][j] = 0.0;
	
	#pragma omp task firstprivate(ca)
	calculate_partial_sp_matrix(self, ca, threshold);
	ca = ca->next;
	
	for(; ca->next; ca=ca->next)
		#pragma omp task firstprivate(ca)
		calculate_sp_matrix(self, ca);
	}
}

{% endif %}

// Functions for normal Lyapunov exponents
{% if n_basic != n: %}

double norm_sq_interval(anchor const v, unsigned int const begin)
{
	anchor const w = *(v.next);
	double const * const vector[4] = {
				&(v.state[begin]), // a
				&(v.diff [begin]), // b/q
				&(w.state[begin]), // c
				&(w.diff [begin])  // d/q
			};
	
	double sum = 0;
	
	for (unsigned int i=0; i<4; i++)
		for (unsigned int j=0; j<4; j++)
			for (unsigned int index=0; index<{{n_basic}}; index++)
				sum += v.sp_matrix[i][j] * vector[i][index] * vector[j][index];
	
	return sum;
}

double norm_sq(dde_integrator const * const self, unsigned int const begin)
{
	double sum = 0;
	#pragma omp parallel
	#pragma omp single
	for (anchor * ca = self->first_anchor; ca->next; ca = ca->next)
		#pragma omp task firstprivate(ca)
		#pragma omp atomic update
		sum += norm_sq_interval(*ca, begin);
	
	return sum;
}

double scalar_product_interval(
	anchor const v,
	unsigned int const begin_1,
	unsigned int const begin_2)
{
	anchor const w = *(v.next);
	double const * const vector_1[4] = {
				&(v.state[begin_1]), // a_1
				&(v.diff [begin_1]), // b_1/q
				&(w.state[begin_1]), // c_1
				&(w.diff [begin_1])  // d_1/q
			};
	
	double const * const vector_2[4] = {
				&(v.state[begin_2]), // a_2
				&(v.diff [begin_2]), // b_2/q
				&(w.state[begin_2]), // c_2
				&(w.diff [begin_2])  // d_2/q
			};
	
	double sum = 0;
	
	for (unsigned int i=0; i<4; i++)
		for (unsigned int j=0; j<4; j++)
			for (unsigned int index=0; index<{{n_basic}}; index++)
				sum += v.sp_matrix[i][j] * vector_1[i][index] * vector_2[j][index];
	
	return sum;
}

double scalar_product(
	dde_integrator const * const self,
	unsigned int const begin_1,
	unsigned int const begin_2)
{
	double sum = 0;
	#pragma omp parallel
	#pragma omp single
	for (anchor * ca = self->first_anchor; ca->next; ca = ca->next)
		#pragma omp task firstprivate(ca)
		#pragma omp atomic update
		sum += scalar_product_interval(*ca, begin_1, begin_2);
	
	return sum;
}

void scale_past(
	dde_integrator const * const self,
	unsigned int const begin,
	double const factor)
{
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		for (unsigned int i=0; i<{{n_basic}}; i++)
		{
			ca->state[begin+i] *= factor;
			ca->diff [begin+i] *= factor;
		}
}

void subtract_from_past(
	dde_integrator const * const self,
	unsigned int const begin_1,
	unsigned int const begin_2,
	double const factor)
{
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		for (unsigned int i=0; i<{{n_basic}}; i++)
		{
			ca->state[begin_1+i] -= factor*ca->state[begin_2+i];
			ca->diff [begin_1+i] -= factor*ca->diff [begin_2+i];
		}
}

static PyObject * orthonormalise(dde_integrator const * const self, PyObject * args)
{
	assert(self->last_anchor);
	assert(self->first_anchor);
	
	unsigned int n_lyap;
	double delay;
	if (!PyArg_ParseTuple(args, "Id", &n_lyap, &delay))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	calculate_sp_matrices(self, delay);
	
	npy_intp dims[1] = { n_lyap };
	PyArrayObject * norms = (PyArrayObject *)PyArray_SimpleNew(1, dims, TYPE_INDEX);
	
	for (unsigned int i=0; i<n_lyap; i++)
	{
		for (unsigned int j=0; j<i; j++)
		{
			double sp = scalar_product(self, (i+1)*{{n_basic}}, (j+1)*{{n_basic}});
			subtract_from_past(self, (i+1)*{{n_basic}}, (j+1)*{{n_basic}}, sp);
		}
		double norm = sqrt(norm_sq(self, (i+1)*{{n_basic}}));
		if (norm > NORM_THRESHOLD)
			scale_past(self, (i+1)*{{n_basic}}, 1./norm);
		* (double *) PyArray_GETPTR1(norms, i) = norm;
	}
	
	return (PyObject *) norms;
}

unsigned int get_dummy(unsigned int const index)
{
	return (2+index)*{{n_basic}};
}

static PyObject * remove_projections(dde_integrator const * const self, PyObject * args)
{
	double delay;
	PyObject * vectors;
	if (!PyArg_ParseTuple(args, "dO!", &delay, &PyList_Type, &vectors))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	calculate_sp_matrices(self, delay);
	
	unsigned int const sep_func = {{n_basic}};
	
	Py_ssize_t len_vectors = PyList_Size(vectors);
	Py_ssize_t d = 2*len_vectors;
	unsigned int dummy_num = 0;
	Py_ssize_t len_dummies = 0;
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
	{
		for (Py_ssize_t vi=0; vi<len_vectors; vi++)
		{
			unsigned int const dummy = get_dummy(dummy_num);
			assert(dummy<{{n}});
			
			PyObject * vector = PyList_GetItem(vectors,vi);
			PyArrayObject * pystate;
			PyArrayObject * pydiff;
			if (!PyArg_ParseTuple(vector, "O!O!", &PyArray_Type, &pystate, &PyArray_Type, &pydiff))
			{
				PyErr_SetString(PyExc_ValueError,"Wrong input.");
				return 0;
			}
			
			if ( (PyArray_TYPE(pystate) != NPY_DOUBLE) || (PyArray_TYPE(pydiff) != NPY_DOUBLE) )
			{
				PyErr_SetString(PyExc_ValueError,"Vectors must be float arrays.");
				return 0;
			}
			
			for (unsigned int j=0; j<{{n_basic}}; j++)
			{
				assert(dummy+j<{{n}});
				for (anchor * oa = self->first_anchor; oa; oa = oa->next)
				{
					oa->state[dummy+j] = 0.0;
					oa->diff [dummy+j] = 0.0;
				}
				ca->state[dummy+j] = * (double *) PyArray_GETPTR1(pystate,j);
				ca->diff [dummy+j] = * (double *) PyArray_GETPTR1(pydiff ,j);
			}
			
			for (unsigned int i=0; i<len_dummies; i++)
			{
				unsigned int const past_dummy = get_dummy((dummy_num-i-1) % d);
				assert(past_dummy<{{n}});
				double const sp = scalar_product(self, dummy, past_dummy);
				subtract_from_past(self, dummy, past_dummy, sp);
			}
			
			double norm = sqrt(norm_sq(self, dummy));
			if (norm > NORM_THRESHOLD)
			{
				scale_past(self, dummy, 1./norm);
				
				double const sp = scalar_product(self, sep_func, dummy);
				subtract_from_past(self, sep_func, dummy, sp);
			}
			else
				scale_past(self, dummy, 0);
			
			len_dummies++;
			dummy_num = (dummy_num+1)%d;
		}
		
		if (len_dummies > len_vectors)
			len_dummies -= len_vectors;
	}
	
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		for (unsigned int j=2*{{n_basic}}; j<{{n}}; j++)
			ca->state[j] = ca->diff[j] = 0.0;
	
	double const norm = sqrt(norm_sq(self, sep_func));
	scale_past(self, sep_func, 1./norm);
	
	return PyFloat_FromDouble(norm);
}

static PyObject * remove_state_component(dde_integrator const * const self, PyObject * args)
{
	unsigned int index;
	if (!PyArg_ParseTuple(args, "I", &index))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		ca->state[{{n_basic}}+index] = 0.0;
	
	Py_RETURN_NONE;
}

static PyObject * remove_diff_component(dde_integrator const * const self, PyObject * args)
{
	unsigned int index;
	if (!PyArg_ParseTuple(args, "I", &index))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		ca->diff[{{n_basic}}+index] = 0.0;
	
	Py_RETURN_NONE;
}

{% endif %}

// Functions for transversal Lyapunov exponents
{% if tangent_indices: %}

unsigned int const tangent_indices[ {{tangent_indices|length}} ] = {
	{% for index in tangent_indices %}
		{{index}} ,
	{% endfor %}
};

double norm_sq_interval_tangent(anchor const v)
{
	anchor const w = *(v.next);
	double const * const vector[4] = {
				v.state, // a
				v.diff , // b/q
				w.state, // c
				w.diff   // d/q
			};
	
	double sum = 0;
	
	for (unsigned int i=0; i<4; i++)
		for (unsigned int j=0; j<4; j++)
			for (unsigned int ti_index=0; ti_index<{{tangent_indices|length}}; ti_index++)
			{
				unsigned int const index = tangent_indices[ti_index];
				sum += v.sp_matrix[i][j] * vector[i][index] * vector[j][index];
			}
	
	return sum;
}

double norm_sq_tangent(dde_integrator const * const self)
{
	double sum = 0;
	#pragma omp parallel
	#pragma omp single
	for (anchor * ca = self->first_anchor; ca->next; ca = ca->next)
		#pragma omp task firstprivate(ca)
		#pragma omp atomic update
		sum += norm_sq_interval_tangent(*ca);
	
	return sum;
}

void scale_past_tangent(
	dde_integrator const * const self,
	double const factor)
{
	for (anchor * ca = self->first_anchor; ca; ca = ca->next)
		for (unsigned int ti_index=0; ti_index<{{tangent_indices|length}}; ti_index++)
		{
			unsigned int const index = tangent_indices[ti_index];
			ca->state[index] *= factor;
			ca->diff [index] *= factor;
		}
}

static PyObject * normalise_indices(dde_integrator const * const self, PyObject * args)
{
	assert(self->last_anchor);
	assert(self->first_anchor);
	
	double delay;
	if (!PyArg_ParseTuple(args, "d", &delay))
	{
		PyErr_SetString(PyExc_ValueError,"Wrong input.");
		return NULL;
	}
	
	calculate_sp_matrices(self,delay);
	
	double norm = sqrt(norm_sq_tangent(self));
	if (norm > NORM_THRESHOLD)
		scale_past_tangent(self,1./norm);
	
	return PyFloat_FromDouble(norm);
}

{% endif %}

// ======================================================

static PyMemberDef dde_integrator_members[] = {
 	{"past_within_step", T_DOUBLE, offsetof(dde_integrator, past_within_step), 0, "past_within_step"},
	{NULL}  /* Sentinel */
};

static PyMethodDef dde_integrator_methods[] = {
	{"get_t", (PyCFunction) get_t, METH_NOARGS, NULL},
	{% if control_pars|length %}
	{"set_parameters", (PyCFunction) set_parameters, METH_VARARGS, NULL},
	{% endif %}
	{"get_recent_state", (PyCFunction) get_recent_state, METH_VARARGS, NULL},
	{"get_next_step", (PyCFunction) get_next_step, METH_VARARGS, NULL},
	{"get_current_state", (PyCFunction) get_current_state, METH_NOARGS, NULL},
	{"get_full_state", (PyCFunction) get_full_state, METH_NOARGS, NULL},
	{"get_p", (PyCFunction) get_p, METH_VARARGS, NULL},
	{"check_new_y_diff", (PyCFunction) check_new_y_diff, METH_VARARGS, NULL},
	{"accept_step", (PyCFunction) accept_step, METH_NOARGS, NULL},
	{"adjust_diff", (PyCFunction) adjust_diff, METH_VARARGS, NULL},
	{"forget", (PyCFunction) forget, METH_VARARGS, NULL},
	{% if n_basic != n: %}
	{"orthonormalise", (PyCFunction) orthonormalise, METH_VARARGS, NULL},
	{"remove_projections", (PyCFunction) remove_projections, METH_VARARGS, NULL},
	{"remove_state_component", (PyCFunction) remove_state_component, METH_VARARGS, NULL},
	{"remove_diff_component", (PyCFunction) remove_diff_component, METH_VARARGS, NULL},
	{% endif %}
	{% if tangent_indices %}
	{"normalise_indices", (PyCFunction) normalise_indices, METH_VARARGS, NULL},
	{% endif %}
	{NULL, NULL, 0, NULL}
};


static PyTypeObject dde_integrator_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
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
	0                          // tp_new
};

static PyMethodDef {{module_name}}_methods[] = {
{NULL, NULL, 0, NULL}
};

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
	
	if (module == NULL)
		return NULL;
	
	Py_INCREF(&dde_integrator_type);
	PyModule_AddObject(module, "dde_integrator", (PyObject *)&dde_integrator_type);
	
	import_array();
	
	return module;
}
