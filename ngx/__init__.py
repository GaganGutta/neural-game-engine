"""ngx -- a playable neural game engine.

Train an action-conditioned world model on a game, then play inside the model.
There is no game engine at inference time: the network predicts the next frame
from the last N frames plus whatever key you just pressed.
"""

__version__ = "0.1.0"
