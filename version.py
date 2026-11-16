
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Square(Rectangle):
    def __init__(self, side_length):
        super().__init__(side_length, side_length)

square1 = Square(5)
print(f"Area of square: {square1.area()}")
# shape = Shape()  # This will raise an error

#This is going to be a comment because we cannot instantiate an abstract class