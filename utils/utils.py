import sys, time
from random import randint

import torch
import numpy as np


class TensorQueue:

    def __init__(self, max_size : int):
        """

        If `queue` reaches max size, every push will pop the oldest items

        :param max_size:
        """
        self.q = torch.Tensor([])
        self.max_size = max_size

    def __len__(self):
        return self.q.shape[-1]

    def push(self,x: torch.Tensor) -> torch.Tensor:
        """Push one or more items

        If Queue is full, the oldest items are poped.

        :param x:
        :return:
        """
        x = torch.as_tensor(x)

        len_before = len(self)
        # cat time dimension
        self.q = torch.cat((self.q,
                            x[-self.max_size:]), -1)

        if len(self) > self.max_size:
            return self.pop(len(self) - self.max_size)
        else:
            return torch.Tensor([])

    def pop(self, num: int = 1) -> torch.Tensor:
        """Pop `num` items

        :param num: How many items to pop. If `num` >= `self.max_size`, everything is poped
        :return: list
        """
        # pop from time dimension
        to_pop = self.q[..., :num]
        self.q = self.q[..., num:]
        return to_pop

    @property
    def queue(self):
        return self.q


def random_chunks(lst: list, min_chunk_size: int, max_chunk_size: int) -> list:

    chunks, i, j = [], 0, 0

    while i <= len(lst) - max_chunk_size:
        j = randint(min_chunk_size, min(max_chunk_size, len(lst)-min_chunk_size - i))
        chunks.append(lst[i:i+j])
        i += j

    chunks.append(lst[i:])
    return chunks


class Progbar(object):
  """Displays a progress bar.
  Arguments:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over time. Metrics in this list
          will be displayed as-is. All others will be averaged
          by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually "step" or "sample").
  """

  def __init__(self, target, width=30, verbose=1, interval=0.05,
               stateful_metrics=None, unit_name='step'):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    self.unit_name = unit_name
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                              sys.stdout.isatty()) or
                             'ipykernel' in sys.modules or
                             'posix' in sys.modules)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0

  def update(self, current, values=None):
    """Updates the progress bar.
    Arguments:
        current: Index of current step.
        values: List of tuples:
            `(name, value_for_last_step)`.
            If `name` is in `stateful_metrics`,
            `value_for_last_step` will be displayed as-is.
            Else, an average of the metric over time will be displayed.
    """
    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        if k not in self._values:
          self._values[k] = [v * (current - self._seen_so_far),
                             current - self._seen_so_far]
        else:
          self._values[k][0] += v * (current - self._seen_so_far)
          self._values[k][1] += (current - self._seen_so_far)
      else:
        # Stateful metrics output a numeric value. This representation
        # means "take an average from a single value" but keeps the
        # numeric formatting.
        self._values[k] = [v, 1]
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if (now - self._last_update < self.interval and
          self.target is not None and current < self.target):
        return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.log10(self.target)) + 1
        bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current

      self._total_width = len(bar)
      sys.stdout.write(bar)

      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600,
                                         (eta % 3600) // 60,
                                         eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))

      if self.target is not None and current >= self.target:
        info += '\n'

      sys.stdout.write(info)
      sys.stdout.flush()

    elif self.verbose == 2:
      if self.target is not None and current >= self.target:
        numdigits = int(np.log10(self.target)) + 1
        count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()

    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)

