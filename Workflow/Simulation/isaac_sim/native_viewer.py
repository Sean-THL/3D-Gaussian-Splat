"""Interactive native PhysX view. All physics/UI calls stay on Kit's main thread.

UI and localhost test/control requests enqueue commands between control steps.
World.render() refreshes graphics without advancing physics in Isaac 4.5.
"""
import hashlib
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import omni.ui as ui
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom, UsdLux


class NativeViewer:
    def __init__(self, sim, application, output, render_client=None, port=8768):
        self.sim, self.application, self.output = sim, application, Path(output)
        self.render_client = render_client
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.latest = {}
        self.quit = False
        self.frames = 0
        self.frame_hash = None
        self.error = None
        self.saved_episodes = set()
        self.last_episode = None
        self.last_camera_time = -1.
        self.clock_error_max = 0.
        stage = sim.world.stage
        self.root = '/World/NativeView'
        UsdGeom.Xform.Define(stage, self.root)
        UsdLux.DomeLight.Define(stage, self.root+'/Light').CreateIntensityAttr(800.)
        self.reference = self.make_mesh('Reference', [.25, .55, .95])
        self.history = self.make_mesh('ActualPath', [.2, .9, .55])
        # Visual children have no collision API or additional rigid body.
        model = UsdGeom.Xform.Define(stage, '/World/NativeDrone/Visual')
        model.AddScaleOp().Set(Gf.Vec3d(1/.16, 1/.10, 1/.04))
        for name, size, pos, color in [
            ('ArmX', [.3, .02, .015], [0., 0., 0.], [.15, .2, .25]),
            ('ArmY', [.02, .3, .015], [0., 0., 0.], [.15, .2, .25]),
            ('Nose', [.08, .025, .025], [.13, 0., .02], [1., .4, .05])]:
            box = UsdGeom.Cube.Define(stage, '/World/NativeDrone/Visual/'+name)
            box.CreateSizeAttr(1.)
            box.AddTranslateOp().Set(Gf.Vec3d(*pos));box.AddScaleOp().Set(Gf.Vec3f(*size))
            box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        for i, (x,y) in enumerate([(.15,0),(-.15,0),(0,.15),(0,-.15)]):
            rotor = UsdGeom.Cylinder.Define(stage, f'/World/NativeDrone/Visual/Rotor{i}')
            rotor.CreateRadiusAttr(.045);rotor.CreateHeightAttr(.008)
            rotor.AddTranslateOp().Set(Gf.Vec3d(x,y,.025))
            rotor.CreateDisplayColorAttr([Gf.Vec3f(.65,.7,.75)])
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window('DASC Native PhysX: thrust/body-rate NMPC', width=720, height=650)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label('Native PhysX flight - forces and torques drive the drone', height=25)
                self.status = ui.Label('Starting paused...', height=65, word_wrap=True)
                ui.ImageWithProvider(self.provider, height=280)
                self.camera_label = ui.Label('Gaussian RGB: connect the pose renderer to enable images', height=40, word_wrap=True)
                with ui.HStack(height=30):
                    for label, action in [('Play / resume','play'),('Pause','pause'),('Restart route','route'),('Reset','reset')]:
                        ui.Button(label, clicked_fn=lambda action=action: self.commands.put({'action':action}))
                with ui.HStack(height=30):
                    for label, action in [('Hold','hover'),('Disturbance','disturbance'),('Constant yaw','yaw_constant'),('Velocity yaw','yaw_aligned'),('Quit viewer','quit')]:
                        ui.Button(label, clicked_fn=lambda action=action: self.commands.put({'action':action}))
                ui.Label('Use these controls. Isaac timeline playback alone does not start the NMPC loop.\nFigure eight and M finish and hold; Restart route begins another episode.', height=40, word_wrap=True)
        self.server = self.make_server(port)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def make_server(self, port):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def respond(self, value, status=200):
                body=json.dumps(value,allow_nan=False).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def do_GET(self):
                if self.path!='/state':return self.respond({'error':'not found'},404)
                with owner.lock:value=dict(owner.latest)
                self.respond(value)
            def do_POST(self):
                try:
                    if self.path!='/control':return self.respond({'error':'not found'},404)
                    size=int(self.headers.get('Content-Length',0))
                    if not 0<size<=1024:raise ValueError('Invalid request size')
                    value=json.loads(self.rfile.read(size))
                    if value.get('action') not in ('play','pause','route','reset','hover','disturbance','yaw_constant','yaw_aligned','quit'):
                        raise ValueError('Unknown viewer action')
                    owner.commands.put(value);self.respond({'queued':True})
                except (ValueError,TypeError) as error:self.respond({'error':str(error)},400)
        return ThreadingHTTPServer(('127.0.0.1',port),Handler)

    def make_mesh(self,name,color):
        mesh=UsdGeom.Mesh.Define(self.sim.world.stage,self.root+'/'+name)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none);mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return mesh

    def set_path(self,mesh,points):
        points=np.asarray(points,dtype=float)
        if len(points)<2:points=np.vstack([points[0],points[0]])
        start,end=points[:-1],points[1:]
        side=np.cross(end-start,[0.,0.,1.]);norm=np.linalg.norm(side,axis=1)
        side=side/np.maximum(norm[:,None],1e-9)*.008
        vertices=np.stack([start-side,start+side,end+side,end-side],axis=1).reshape(-1,3)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([4]*len(start));mesh.CreateFaceVertexIndicesAttr(list(range(len(vertices))))

    def save_episode(self):
        from Workflow.Simulation.run import write_telemetry
        if self.sim.episode in self.saved_episodes:return
        folder=self.output/f'view_episode_{self.sim.episode:04d}';folder.mkdir(exist_ok=True)
        write_telemetry(folder/'telemetry.csv',self.sim.telemetry)
        (folder/'episode_parameters.json').write_text(json.dumps(self.sim.episode_parameters,indent=2)+'\n')
        with (folder/'demonstrations.jsonl').open('w') as stream:
            for row in self.sim.demonstrations:stream.write(json.dumps(row)+'\n')
        self.saved_episodes.add(self.sim.episode)

    def process_commands(self):
        while not self.commands.empty():
            value=self.commands.get_nowait();action=value['action']
            if action=='quit':self.quit=True;continue
            if action in ('route','reset','yaw_constant','yaw_aligned'):self.save_episode()
            if action.startswith('yaw_'):
                self.sim.command({'action':'configure','task':{'yaw_mode':'constant' if action=='yaw_constant' else 'velocity_aligned'}})
            else:self.sim.command({'action':action})
            self.last_camera_time=-1.

    def publish(self,refresh_camera=False):
        state=self.sim.snapshot()
        if self.last_episode!=self.sim.episode:
            self.set_path(self.reference,state['route'])
            center=np.mean(np.asarray(state['route']),axis=0)
            set_camera_view(eye=center+np.array([2.,-2.5,1.8]),target=center)
            self.last_episode=self.sim.episode
        self.set_path(self.history,state['history'][-800:])
        if self.render_client and refresh_camera:
            rgb,_=self.render_client.render(self.sim.state)
            rgba=np.concatenate([rgb,np.full((*rgb.shape[:2],1),255,dtype=np.uint8)],axis=2)
            self.provider.set_bytes_data(rgba.ravel().tolist(),[rgb.shape[1],rgb.shape[0]])
            self.frames+=1;self.frame_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
            self.camera_label.text=f'Native 3DGUT camera | frames {self.frames} | t={self.sim.time:.2f}s'
        phase='PAUSED' if state['paused'] else 'HOLDING' if state['mode']=='hover' else 'ROUTE COMPLETE - HOLDING' if state['complete'] else 'FLYING'
        self.status.text=(f"{phase} | route {state['route_time_s']:.1f}/{state['duration_s']:.1f}s | episode {state['episode']}\n"
            f"PhysX pose {state['position'][0]:+.2f}, {state['position'][1]:+.2f}, {state['position'][2]:.2f} m | error {state['tracking_error_m']*100:.1f} cm\n"
            f"Thrust {state['thrust_n']:.2f} N | NMPC | {self.error or 'OK'}")
        physics_elapsed=float(self.sim.world.current_time)-self.sim.physics_epoch_time_s
        clock_error=abs(physics_elapsed-self.sim.time)
        self.clock_error_max=max(self.clock_error_max,clock_error)
        state.update(camera_frames=self.frames,camera_sha256=self.frame_hash,viewer_error=self.error,
                     native_physics_time_s=physics_elapsed,physics_clock_error_s=clock_error,
                     max_physics_clock_error_s=self.clock_error_max)
        with self.lock:self.latest=state
        temporary=self.output/'viewer_status.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(self.output/'viewer_status.json')

    def run(self,seconds=0.):
        self.sim.paused=True;self.sim.world.pause()
        start=time.monotonic();last_refresh=-1.
        self.publish(refresh_camera=True)
        print('NATIVE_VIEWER_READY: use the DASC Native PhysX panel, initially paused.',flush=True)
        try:
            while self.application.is_running() and not self.quit and (seconds<=0 or time.monotonic()-start<seconds):
                tick=time.monotonic()
                try:
                    self.process_commands()
                    if not self.sim.paused:
                        if not self.sim.world.is_playing():self.sim.world.play()
                        self.sim.advance()
                    else:self.sim.world.pause()
                    refresh=(self.last_camera_time<0 or abs(self.sim.time-self.last_camera_time)>=.1-1e-8)
                    self.publish(refresh_camera=refresh)
                    if refresh:self.last_camera_time=self.sim.time
                    self.error=None
                except Exception as error:
                    self.error=f'{type(error).__name__}: {error}'
                    self.sim.paused=True;self.sim.world.pause();self.status.text=self.error
                self.sim.world.render()  # Isaac 4.5 explicitly disables simulation during this update.
                time.sleep(max(0.,self.sim.control_dt-(time.monotonic()-tick)))
        except KeyboardInterrupt:
            pass
        finally:
            self.sim.paused=True;self.sim.world.pause();self.save_episode()
            self.server.shutdown();self.server.server_close()
        return dict(status='closed',camera_frames=self.frames,max_physics_clock_error_s=self.clock_error_max,
                    last_episode=self.sim.episode,simulation_time_s=self.sim.time,error=self.error)
