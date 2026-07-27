# def http_error(status):
#     match status:
#         case 400:
#             return 'bad request'
#         case 401:
#             return 'unauthorized'
#         case 403:
#             return 'forbidden'
#         case 404:
#             return 'Not found'
#         case _:
#             return 'Internal error'
# res=http_error(404)
# print(res)
class Point:
    __match_args__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def where_is_points(points):
    match points:
        case Point(0, 0):
            print("The origin")
        case Point(x, y) if x != 0 and y != 0:
            print(f"Single point {x}, {y}")
        case Point(0, y1):
            print(f"One on the Y axis at {y1}")
        case _:
            print("Something else")


where_is_points(Point(0, 0))
where_is_points(Point(1, 1))
where_is_points(Point(0, 1))
