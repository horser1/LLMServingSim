import re
from .logger import get_logger

class Controller():
    def __init__(self, total_num, telemetry=None):
        self.end_dict = {}
        self.total_num = total_num
        self.telemetry = telemetry
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    def read_wait(self, p):
        start = None
        if self.telemetry is not None:
            from time import perf_counter_ns
            start = perf_counter_ns()
        out = [""]
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            # For debugging
            # print(line, end='')
            out.append(line)
            p.stdout.flush()
        if start is not None:
            self.telemetry.phase_ns["astra_ipc_wait"] += perf_counter_ns() - start
            self.telemetry.phase_calls["astra_ipc_wait"] += 1
        return out

    def check_end(self, p):
        out = ["",""]
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            out.append(p.stdout.readline())
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        start = None
        if self.telemetry is not None:
            from time import perf_counter_ns
            start = perf_counter_ns()
        # For debugging
        # print(input)
        p.stdin.write(input+'\n')
        p.stdin.flush()
        if start is not None:
            self.telemetry.phase_ns["astra_ipc_write"] += perf_counter_ns() - start
            self.telemetry.phase_calls["astra_ipc_write"] += 1
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return
