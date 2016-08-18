from __future__ import print_function

import sympy
import unittest
from jitcdde._helpers import *

class TestCollectArguments(unittest.TestCase):
	def test_complex_expression(self):
		f = sympy.Function("f")
		g = sympy.Function("g")
		a = sympy.Symbol("a")
		
		expression = 3**f(42) + 23 - f(43,44) + f(45+a)*sympy.sin( g(f(46,47,48)+17) - g(4) )
		
		self.assertEqual(
			set(collect_arguments(expression, f)),
			{
				(sympy.Integer(42),),
				(sympy.Integer(43), sympy.Integer(44)),
				(45+a,),
				(sympy.Integer(46), sympy.Integer(47), sympy.Integer(48))}
			)

unittest.main(buffer=True)
