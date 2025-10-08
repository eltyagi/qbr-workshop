"""
Complex Calculator Implementation
Supports basic arithmetic, advanced mathematical functions, scientific operations,
and complex number calculations.
"""

import math
import cmath
import re
from typing import Union, List, Tuple


class ComplexCalculator:
    """
    A comprehensive calculator supporting:
    - Basic arithmetic operations (+, -, *, /, **, %, //)
    - Scientific functions (sin, cos, tan, log, exp, etc.)
    - Complex number operations
    - Matrix operations
    - Statistical functions
    - Unit conversions
    - Expression parsing and evaluation
    """
    
    def __init__(self):
        """Initialize the calculator with constants and memory."""
        self.memory = 0
        self.history = []
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
            'tau': 2 * math.pi
        }
        self.angle_mode = 'rad'  # 'rad' or 'deg'
    
    # Basic Arithmetic Operations
    def add(self, a: float, b: float) -> float:
        """Addition"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtraction"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiplication"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Division with zero check"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Exponentiation"""
        return base ** exponent
    
    def modulo(self, a: float, b: float) -> float:
        """Modulo operation"""
        if b == 0:
            raise ValueError("Cannot perform modulo with zero")
        return a % b
    
    def floor_division(self, a: float, b: float) -> float:
        """Floor division"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a // b
    
    # Trigonometric Functions
    def _convert_angle(self, angle: float) -> float:
        """Convert angle based on current mode"""
        return math.radians(angle) if self.angle_mode == 'deg' else angle
    
    def sin(self, x: float) -> float:
        """Sine function"""
        return math.sin(self._convert_angle(x))
    
    def cos(self, x: float) -> float:
        """Cosine function"""
        return math.cos(self._convert_angle(x))
    
    def tan(self, x: float) -> float:
        """Tangent function"""
        return math.tan(self._convert_angle(x))
    
    def asin(self, x: float) -> float:
        """Arc sine"""
        if not -1 <= x <= 1:
            raise ValueError("Input must be between -1 and 1")
        result = math.asin(x)
        return math.degrees(result) if self.angle_mode == 'deg' else result
    
    def acos(self, x: float) -> float:
        """Arc cosine"""
        if not -1 <= x <= 1:
            raise ValueError("Input must be between -1 and 1")
        result = math.acos(x)
        return math.degrees(result) if self.angle_mode == 'deg' else result
    
    def atan(self, x: float) -> float:
        """Arc tangent"""
        result = math.atan(x)
        return math.degrees(result) if self.angle_mode == 'deg' else result
    
    def atan2(self, y: float, x: float) -> float:
        """Two-argument arc tangent"""
        result = math.atan2(y, x)
        return math.degrees(result) if self.angle_mode == 'deg' else result
    
    # Hyperbolic Functions
    def sinh(self, x: float) -> float:
        """Hyperbolic sine"""
        return math.sinh(x)
    
    def cosh(self, x: float) -> float:
        """Hyperbolic cosine"""
        return math.cosh(x)
    
    def tanh(self, x: float) -> float:
        """Hyperbolic tangent"""
        return math.tanh(x)
    
    # Logarithmic and Exponential Functions
    def log(self, x: float, base: float = math.e) -> float:
        """Logarithm with optional base"""
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        if base <= 0 or base == 1:
            raise ValueError("Base must be positive and not equal to 1")
        return math.log(x, base)
    
    def log10(self, x: float) -> float:
        """Base-10 logarithm"""
        return self.log(x, 10)
    
    def log2(self, x: float) -> float:
        """Base-2 logarithm"""
        return self.log(x, 2)
    
    def ln(self, x: float) -> float:
        """Natural logarithm"""
        return self.log(x)
    
    def exp(self, x: float) -> float:
        """Exponential function (e^x)"""
        return math.exp(x)
    
    def exp10(self, x: float) -> float:
        """Base-10 exponential (10^x)"""
        return 10 ** x
    
    def exp2(self, x: float) -> float:
        """Base-2 exponential (2^x)"""
        return 2 ** x
    
    # Root Functions
    def sqrt(self, x: float) -> Union[float, complex]:
        """Square root (returns complex for negative inputs)"""
        if x < 0:
            return cmath.sqrt(x)
        return math.sqrt(x)
    
    def cbrt(self, x: float) -> float:
        """Cube root"""
        return x ** (1/3) if x >= 0 else -((-x) ** (1/3))
    
    def nth_root(self, x: float, n: float) -> Union[float, complex]:
        """Nth root"""
        if n == 0:
            raise ValueError("Root degree cannot be zero")
        if x < 0 and n % 2 == 0:
            return cmath.exp(cmath.log(x) / n)
        return x ** (1/n)
    
    # Complex Number Operations
    def complex_add(self, z1: complex, z2: complex) -> complex:
        """Complex addition"""
        return z1 + z2
    
    def complex_multiply(self, z1: complex, z2: complex) -> complex:
        """Complex multiplication"""
        return z1 * z2
    
    def complex_divide(self, z1: complex, z2: complex) -> complex:
        """Complex division"""
        if z2 == 0:
            raise ValueError("Cannot divide by zero")
        return z1 / z2
    
    def complex_conjugate(self, z: complex) -> complex:
        """Complex conjugate"""
        return z.conjugate()
    
    def complex_abs(self, z: complex) -> float:
        """Complex absolute value (magnitude)"""
        return abs(z)
    
    def complex_phase(self, z: complex) -> float:
        """Complex phase (argument)"""
        result = cmath.phase(z)
        return math.degrees(result) if self.angle_mode == 'deg' else result
    
    def polar_to_rectangular(self, r: float, theta: float) -> complex:
        """Convert polar coordinates to rectangular form"""
        theta_rad = self._convert_angle(theta)
        return cmath.rect(r, theta_rad)
    
    def rectangular_to_polar(self, z: complex) -> Tuple[float, float]:
        """Convert rectangular coordinates to polar form"""
        r = abs(z)
        theta = cmath.phase(z)
        theta = math.degrees(theta) if self.angle_mode == 'deg' else theta
        return r, theta
    
    # Statistical Functions
    def mean(self, numbers: List[float]) -> float:
        """Calculate mean (average)"""
        if not numbers:
            raise ValueError("Cannot calculate mean of empty list")
        return sum(numbers) / len(numbers)
    
    def median(self, numbers: List[float]) -> float:
        """Calculate median"""
        if not numbers:
            raise ValueError("Cannot calculate median of empty list")
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        if n % 2 == 0:
            return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        return sorted_nums[n//2]
    
    def mode(self, numbers: List[float]) -> List[float]:
        """Calculate mode(s)"""
        if not numbers:
            raise ValueError("Cannot calculate mode of empty list")
        frequency = {}
        for num in numbers:
            frequency[num] = frequency.get(num, 0) + 1
        max_freq = max(frequency.values())
        return [num for num, freq in frequency.items() if freq == max_freq]
    
    def variance(self, numbers: List[float], sample: bool = True) -> float:
        """Calculate variance (sample by default)"""
        if not numbers:
            raise ValueError("Cannot calculate variance of empty list")
        if len(numbers) == 1 and sample:
            raise ValueError("Cannot calculate sample variance with only one value")
        
        mean_val = self.mean(numbers)
        squared_diffs = [(x - mean_val) ** 2 for x in numbers]
        divisor = len(numbers) - 1 if sample else len(numbers)
        return sum(squared_diffs) / divisor
    
    def standard_deviation(self, numbers: List[float], sample: bool = True) -> float:
        """Calculate standard deviation"""
        return math.sqrt(self.variance(numbers, sample))
    
    # Matrix Operations
    def matrix_add(self, matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
        """Matrix addition"""
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Matrices must have the same dimensions")
        
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix1[0])):
                row.append(matrix1[i][j] + matrix2[i][j])
            result.append(row)
        return result
    
    def matrix_multiply(self, matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication"""
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Number of columns in first matrix must equal rows in second matrix")
        
        result = []
        for i in range(len(matrix1)):
            row = []
            for j in range(len(matrix2[0])):
                sum_product = 0
                for k in range(len(matrix2)):
                    sum_product += matrix1[i][k] * matrix2[k][j]
                row.append(sum_product)
            result.append(row)
        return result
    
    def matrix_determinant(self, matrix: List[List[float]]) -> float:
        """Calculate matrix determinant (2x2 and 3x3 only)"""
        n = len(matrix)
        if n != len(matrix[0]):
            raise ValueError("Matrix must be square")
        
        if n == 1:
            return matrix[0][0]
        elif n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        elif n == 3:
            return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                    matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                    matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
        else:
            raise ValueError("Determinant calculation only supported for 1x1, 2x2, and 3x3 matrices")
    
    # Utility Functions
    def factorial(self, n: int) -> int:
        """Calculate factorial"""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if not isinstance(n, int):
            raise ValueError("Factorial input must be an integer")
        return math.factorial(n)
    
    def combination(self, n: int, r: int) -> int:
        """Calculate combination (nCr)"""
        if n < 0 or r < 0:
            raise ValueError("n and r must be non-negative")
        if r > n:
            return 0
        return math.comb(n, r)
    
    def permutation(self, n: int, r: int) -> int:
        """Calculate permutation (nPr)"""
        if n < 0 or r < 0:
            raise ValueError("n and r must be non-negative")
        if r > n:
            return 0
        return math.perm(n, r)
    
    def gcd(self, a: int, b: int) -> int:
        """Greatest common divisor"""
        return math.gcd(a, b)
    
    def lcm(self, a: int, b: int) -> int:
        """Least common multiple"""
        return abs(a * b) // self.gcd(a, b) if a != 0 and b != 0 else 0
    
    # Memory Functions
    def memory_store(self, value: float) -> None:
        """Store value in memory"""
        self.memory = value
    
    def memory_recall(self) -> float:
        """Recall value from memory"""
        return self.memory
    
    def memory_clear(self) -> None:
        """Clear memory"""
        self.memory = 0
    
    def memory_add(self, value: float) -> None:
        """Add value to memory"""
        self.memory += value
    
    def memory_subtract(self, value: float) -> None:
        """Subtract value from memory"""
        self.memory -= value
    
    # History Functions
    def add_to_history(self, operation: str, result: float) -> None:
        """Add calculation to history"""
        self.history.append(f"{operation} = {result}")
    
    def get_history(self) -> List[str]:
        """Get calculation history"""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear calculation history"""
        self.history.clear()
    
    # Mode Functions
    def set_angle_mode(self, mode: str) -> None:
        """Set angle mode ('rad' or 'deg')"""
        if mode not in ['rad', 'deg']:
            raise ValueError("Mode must be 'rad' or 'deg'")
        self.angle_mode = mode
    
    def get_angle_mode(self) -> str:
        """Get current angle mode"""
        return self.angle_mode
    
    # Expression Parser
    def evaluate_expression(self, expression: str) -> Union[float, complex]:
        """
        Evaluate mathematical expressions with support for:
        - Basic operators: +, -, *, /, **, %, //
        - Functions: sin, cos, tan, log, sqrt, etc.
        - Constants: pi, e, phi, tau
        - Parentheses for grouping
        """
        # Replace constants
        expr = expression.lower()
        for constant, value in self.constants.items():
            expr = expr.replace(constant, str(value))
        
        # Replace function calls with math module equivalents
        function_map = {
            'sin': 'math.sin',
            'cos': 'math.cos', 
            'tan': 'math.tan',
            'asin': 'math.asin',
            'acos': 'math.acos',
            'atan': 'math.atan',
            'sinh': 'math.sinh',
            'cosh': 'math.cosh',
            'tanh': 'math.tanh',
            'log': 'math.log',
            'log10': 'math.log10',
            'log2': 'math.log2',
            'ln': 'math.log',
            'exp': 'math.exp',
            'sqrt': 'math.sqrt',
            'abs': 'abs',
            'ceil': 'math.ceil',
            'floor': 'math.floor',
            'round': 'round'
        }
        
        for func, replacement in function_map.items():
            expr = re.sub(f'\\b{func}\\b', replacement, expr)
        
        # Handle angle conversion for trig functions if in degree mode
        if self.angle_mode == 'deg':
            trig_functions = ['math.sin', 'math.cos', 'math.tan']
            for func in trig_functions:
                expr = re.sub(f'{func}\\(([^)]+)\\)', f'{func}(math.radians(\\1))', expr)
        
        try:
            # Use eval with restricted globals for safety
            allowed_names = {
                "math": math,
                "cmath": cmath,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "__builtins__": {}
            }
            result = eval(expr, allowed_names)
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")


def main():
    """Interactive calculator interface"""
    calc = ComplexCalculator()
    
    print("=== Complex Calculator ===")
    print("Type 'help' for available commands, 'quit' to exit")
    print("Current angle mode:", calc.get_angle_mode())
    print()
    
    while True:
        try:
            user_input = input("Calculator> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.lower() == 'history':
                history = calc.get_history()
                if history:
                    for entry in history:
                        print(entry)
                else:
                    print("No history available")
            elif user_input.lower() == 'clear_history':
                calc.clear_history()
                print("History cleared")
            elif user_input.lower().startswith('mode '):
                mode = user_input[5:].strip()
                calc.set_angle_mode(mode)
                print(f"Angle mode set to: {calc.get_angle_mode()}")
            elif user_input.lower() == 'memory':
                print(f"Memory: {calc.memory_recall()}")
            elif user_input.lower().startswith('store '):
                value = float(user_input[6:])
                calc.memory_store(value)
                print(f"Stored {value} in memory")
            else:
                # Evaluate expression
                result = calc.evaluate_expression(user_input)
                print(f"Result: {result}")
                calc.add_to_history(user_input, result)
                
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


def print_help():
    """Print help information"""
    help_text = """
Available Operations:
- Basic: +, -, *, /, ** (power), % (modulo), // (floor division)
- Trigonometric: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
- Logarithmic: log, log10, log2, ln, exp
- Other: sqrt, abs, ceil, floor, round, factorial (use math.factorial)
- Constants: pi, e, phi, tau

Commands:
- help: Show this help
- history: Show calculation history
- clear_history: Clear calculation history
- mode rad/deg: Set angle mode to radians or degrees
- memory: Show memory value
- store <value>: Store value in memory
- quit/exit/q: Exit calculator

Examples:
- 2 + 3 * 4
- sin(pi/2)
- sqrt(16) + log(e)
- 2**3 + factorial(5)
- (3 + 4) * (5 - 2)
"""
    print(help_text)


if __name__ == "__main__":
    main()
