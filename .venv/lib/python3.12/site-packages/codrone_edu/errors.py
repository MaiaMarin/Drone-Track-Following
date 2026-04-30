# Start of Color Function Errors
class ColorProcessingError(Exception):
    """Generalized Exception raised for color training/detection functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class LoadColorsetError(ColorProcessingError):
    """Exception raised for load_color_data() function."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


class DetectColorError(ColorProcessingError):
    """Exception raised for color detection function."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


class FolderNotFoundError(ColorProcessingError):
    """Exception raised for colorset print/append functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class ColorsetNotFoundError(ColorProcessingError):
    """Exception raised for colorset print/append functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class ColorNotFoundError(ColorProcessingError):
    """Exception raised for colorset print/append functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message
# End of Color Function Errors

# Start of Screen Drawing Function Errors
class ScreenDrawError(Exception):
    """Generalized Exception raised for screen drawing functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidDrawColorError(ScreenDrawError):
    """Exception raised for invalid color passed in screen drawing functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidPointListError(ScreenDrawError):
    """Exception raised for invalid point lists used in polygon, chord, and ellipse drawings."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidImageError(ScreenDrawError):
    """Exception raised for invalid image object passed in screen drawing functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidAlignmentError(ScreenDrawError):
    """Exception raised for invalid alignment value or invalid x-start/-end in controller_draw_string_align() functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidPixelListError(ScreenDrawError):
    """Exception raised for invalid pixel list in controller_draw_image() functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message
# End of Screen Drawing Function Errors


# Start of Swarm Function Errors
class SwarmError(Exception):
    """Generalized Exception raised for swarm functions."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidSyncError(SwarmError):
    """Exception raised for invalid sync object in swarm.run()."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message

class InvalidOrderError(SwarmError):
    """Exception raised for invalid 'order' value in swarm.run()."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message
# End of Swarm Function Errors
