#!/usr/bin/env python
"""Run Phenix's development Xtrapol8 command with a fixed FOBS label lookup.

The Phenix 2.0-5936 dispatcher asks DataManager for arrays labelled ``FOBS``.
For MTZ files with an amplitude/sigma pair, DataManager expects the combined
label string ``FOBS,SIGFOBS``. This launcher keeps the bundled ProgramTemplate
workflow intact while patching only that lookup.
"""

from __future__ import absolute_import, division, print_function

import sys

from iotbx.cli_parser import run_program
from mmtbx.maps import xtrapol8 as xtrapol8_maps
from mmtbx.programs import xtrapol8 as xtrapol8_program


class PatchedProgram(xtrapol8_program.Program):
  def run(self):
    print(
      "Using model file:",
      self.data_manager.get_default_model_name(),
      file=self.logger)
    print(
      "Using reflection file(s):",
      self.data_manager.get_miller_array_names(),
      file=self.logger)

    model_reference = self.data_manager.get_model()
    hkls = self.data_manager.get_miller_array_names()
    fn_reference = hkls[0]
    fn_triggered = hkls[1]
    f_obs_reference = self.data_manager.get_miller_arrays(
      ["FOBS,SIGFOBS"], fn_reference)[0]
    f_obs_triggered = self.data_manager.get_miller_arrays(
      ["FOBS,SIGFOBS"], fn_triggered)[0]

    xtr = xtrapol8_maps.manager(
      model_reference=model_reference,
      f_obs_reference=f_obs_reference,
      f_obs_triggered=f_obs_triggered,
      log=sys.stdout)
    xtr.run()


if __name__ == "__main__":
  run_program(program_class=PatchedProgram)
