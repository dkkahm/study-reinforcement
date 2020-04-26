from .. import actions
from ..ActionSpace import ActionSpace
from ..ObservationSpace import ObservationSpace


class Environment5x5Moving:
    def __init__(self):
        self.observation_space = ObservationSpace(float('inf'))
        self.action_space = ActionSpace(5)
        self.reset()

    def reset(self):
        self.s = [0, 5, 11, 17]
        self.last_a = None
        self.done = False
        self.d = [1, 1, 1]
        return self.s

    def reset_state(self, s):
        self.last_a = None
        self.s = s
        self.done = Environment5x5Moving._is_done(s)

    def step(self, a):
        if self.done:
            return (self.s, 0, self.done, None)

        for o in range(3):
            orow = self.s[o + 1] // 5
            ocol = self.s[o + 1] % 5
            d = self.d[o]
            # print(o, orow, ocol, d)

            ocol = ocol + d
            # print(ocol)

            if ocol >= 5:
                ocol = 3
                d = -1
            elif ocol < 0:
                ocol = 1
                d = 1
            # print(ocol, d)

            self.s[o + 1] = orow * 5 + ocol
            self.d[o] = d
            # print(self.s, self.d)

        ston = self.s[0]
        srow = ston // 5
        scol = ston % 5
        # print(self.s, action_name(a), (srow, scol), "=>", end="")

        if a == actions.LEFT:
            if scol > 0:
                scol -= 1
        elif a == actions.DOWN:
            if srow < 4:
                srow += 1
        elif a == actions.RIGHT:
            if scol < 4:
                scol += 1
        elif a == actions.UP:
            if srow > 0:
                srow -= 1
        elif a == actions.NONE:
            pass
        else:
            raise ValueError
        # print((srow, scol))

        self.s[0] = srow * 5 + scol

        self.last_a = a
        self.done = self._is_done()

        return (self.s, self._get_reward(), self.done, None)

    def render(self):
        if self.last_a is not None:
            print(f"({actions.action_name(self.last_a)})")

        ston = self.s[0]

        for row in range(5):
            for col in range(5):
                p = row * 5 + col

                if p == ston:
                    print("X", end="")
                elif self._is_obsticle(p):
                    print("V", end="")
                elif self._is_goal(p):
                    print("G", end="")
                else:
                    print("_", end="")
            print("")
        print("")

    def _is_done(self):
        ston = self.s[0]

        if self._is_goal(ston):
            return True

        return self._is_obsticle(ston)

    def _is_obsticle(self, p):
        for o in range(3):
            if p == self.s[o + 1]:
                return True

        return False

    def _is_goal(self, p):
        return p == 24

    def _get_reward(self):
        ston = self.s[0]

        if self._is_goal(ston):
            return 1

        for o in range(3):
            if ston == self.s[o + 1]:
                return -1

        return 0
