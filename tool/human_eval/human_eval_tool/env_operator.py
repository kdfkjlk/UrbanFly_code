import os
import subprocess
import signal
import sys
import time
import errno



class EventHandler():

    def __init__(self, envs_folder_path=None, agent_folder_path=None):
        self.port = 41451
        self.envs_folder_path = '../DATA/envs' if envs_folder_path is None else envs_folder_path
        self.agent_folder_path = '../Trajectory' if agent_folder_path is None else agent_folder_path

        self.MapName2file_dict = {
            # "ModernCityEnvironment": "ModernCityDay",
        }


    def mapName2shName(self, map_name):
        try:
            return self.MapName2file_dict[map_name]
        except:
            return map_name


    def get_env_file_name(self, env_name):
        file_path = os.path.join(self.envs_folder_path, env_name, env_name + ".sh")
        file_path2 = os.path.join(self.envs_folder_path, env_name, "AirSimEnv.sh")

        if os.path.exists(file_path):
            return file_path
        if os.path.exists(file_path2):
            return file_path2
        else:
            print(f"Path '{file_path}' and '{file_path2}' does not exist.")
            sys.exit(1)



    def run_shell_script(self, script_path):
        raise NotImplementedError(
            "Automatic AirSim launch is not supported in this release. "
            "Please start the AirSim map manually and run with --manual_env."
        )


    def FromPortGetPid(self, port: int):
        subprocess_execute = "netstat -nlp | grep {}".format(
            port,
        )

        try:
            p = subprocess.Popen(
                subprocess_execute,
                stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                shell=True,
            )
        except Exception as e:
            print(
                "{}\t{}\t{}".format(
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                    'FromPortGetPid',
                    e,
                )
            )
            return None
        except:
            return None

        pid = None
        for line in iter(p.stdout.readline, b''):
            line = str(line, encoding="utf-8")
            if 'tcp' in line:
                pid = line.strip().split()[-1].split('/')[0]
                try:
                    pid = int(pid)
                except:
                    pid = None
                break

        try:
            # os.system(("kill -9 {}".format(p.pid)))
            os.kill(p.pid, signal.SIGKILL)
        except:
            pass

        return pid


    def pid_exists(self, pid) -> bool:
        """
        Check whether pid exists in the current process table.
        UNIX only.
        """
        if pid < 0:
            return False

        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:
                # ESRCH == No such process
                return False
            elif err.errno == errno.EPERM:
                # EPERM clearly means there's a process to deny access to
                return True
            else:
                # According to "man 2 kill" possible error values are
                # (EINVAL, EPERM, ESRCH)
                raise
        else:
            return True


    def KillPid(self, pid) -> None:
        if pid is None or not isinstance(pid, int):
            print('pid is not int')
            return

        while self.pid_exists(pid):
            try:
                # os.system(("kill -9 {}".format(pid)))
                os.kill(pid, signal.SIGKILL)
            except Exception as e:
                pass
            time.sleep(0.5)

        return


    def _kill_port(self):
        pid = self.FromPortGetPid(self.port)
        self.KillPid(pid)


    def run_sh_file(self, map_name):
        env_name = self.mapName2shName(map_name=map_name)
        sh_file = self.get_env_file_name(env_name)
        process = self.run_shell_script(sh_file)
        return process


    def check_airsim_process(self):
        # 检查 AirSim 相关进程是否还在运行
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
        processes = result.stdout.decode('utf-8')

        if 'AirSimEnv' in processes:
            print("AirSim process is still running.")
            return True
        else:
            print("No AirSim process running.")
            return False


    def kill_airsim_process(self):
        try:
            # 强制终止所有 AirSim 相关的进程
            subprocess.run(['killall', 'AirSimEnv'])
            os.system('killall AirSimEnv')
            print("Terminated all AirSimEnv processes.")
        except Exception as e:
            print(f"Failed to terminate AirSimEnv processes: {e}")
            sys.exit(1)



    def check_running_sh_file(self):
        if self.check_airsim_process():
            self.kill_airsim_process()
            time.sleep(5)



    def close_sh_file(self):
        self._kill_port()
        if self.check_airsim_process():
            self.kill_airsim_process()
            time.sleep(3)



simulator_handeler = EventHandler()



