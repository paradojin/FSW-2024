import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:async';
import 'dart:convert';
import 'package:audioplayers/audioplayers.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(MyApp(camera: firstCamera));
}

/*void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Inicializar las cámaras disponibles en el dispositivo
  final cameras = await availableCameras();
  
  // Seleccionar la cámara frontal en lugar de la primera cámara disponible
  final frontCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.front,
  );

  runApp(MyApp(camera: frontCamera));
}*/

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  MyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Monitoreo del Conductor en Tiempo Real',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: LoadingScreen(camera: camera),
    );
  }
}

class LoadingScreen extends StatefulWidget {
  final CameraDescription camera;

  LoadingScreen({required this.camera});

  @override
  _LoadingScreenState createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen> {
  bool isConnected = false;
  String connectionMessage = "Verificando la conexión al servidor...";

  @override
  void initState() {
    super.initState();
    _checkConnection();
  }

  Future<void> _checkConnection() async {
  try {
    var response = await http
        .get(Uri.parse('http://10.0.2.2:8000/ping/'))
        .timeout(Duration(seconds: 3)); // Tiempo de espera de 3 segundos

    if (response.statusCode == 200) {
      setState(() {
        isConnected = true;
        connectionMessage = "Conexión exitosa. Redirigiendo...";
      });
      await Future.delayed(Duration(seconds: 2));
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => CameraScreen(camera: widget.camera)),
      );
    } else {
      setState(() {
        connectionMessage = "Error: No se pudo conectar con el servidor.";
      });
      _showConnectionError();
    }
  } catch (e) {
    setState(() {
      connectionMessage = "Error: No se pudo conectar con el servidor.";
    });
    _showConnectionError();
  }
}

  void _showConnectionError() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Error de Conexión'),
          content: Text('No se pudo conectar con el servidor. Verifica tu conexión e inténtalo de nuevo.'),
          actions: [
            TextButton(
              child: Text('Reintentar'),
              onPressed: () {
                Navigator.of(context).pop();
                _checkConnection();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 20),
            Text(connectionMessage, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;

  CameraScreen({required this.camera});

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  late Future<void> _initializeControllerFuture;
  bool isSendingFrames = false;
  Timer? _timer;

  String leftEyeStatus = 'Unknown';
  String rightEyeStatus = 'Unknown';
  double mar = 0.0;
  bool yawnDetected = false;
  double somnolenciaPuntuacion = 0.0;
  int totalBlinks = 0;
  int totalYawns = 0;

  int microsuenosAcumulados= 0;
  double blinkRate60s= 0.0;
  double yawnRate60s= 0.0;

  AudioPlayer? audioPlayer;
  bool isAlertActive = false;
  bool hasAlertTriggered = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    audioPlayer = AudioPlayer();
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.low,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller!.initialize();
    await _initializeControllerFuture;
  }

  void _startOrStopSendingFrames() async {
  if (isSendingFrames) {
    // Si el viaje está activo y se presiona "Finalizar viaje"
    setState(() {
      isSendingFrames = false;
      _controller?.stopImageStream();
    });

    // Enviar la solicitud de finalizar viaje al servidor
    try {
      var response = await http.get(Uri.parse('http://10.0.2.2:8000/end_trip/'));  //http://10.0.2.2:8000/detect/
      if (response.statusCode == 200) {
        // Respuesta exitosa, resetear las variables
        setState(() {
          leftEyeStatus = 'Unknown';
          rightEyeStatus = 'Unknown';
          mar = 0.0;
          yawnDetected = false;
          somnolenciaPuntuacion = 0.0;
          totalBlinks = 0;
          totalYawns = 0;
          isAlertActive = false;
          hasAlertTriggered = false;
          microsuenosAcumulados = 0;
          blinkRate60s = 0.0;
          yawnRate60s = 0.0;
        });
      } else {
        print('Error al finalizar el viaje. Status code: ${response.statusCode}');
      }
    } catch (e) {
      print('Error al enviar la solicitud de finalizar viaje: $e');
    }
  } else {
    // Si el viaje no está activo y se presiona "Iniciar viaje"
    setState(() {
      isSendingFrames = true;
      _controller?.startImageStream((CameraImage image) {
        if (_timer?.isActive ?? false) return;
        _timer = Timer(Duration(milliseconds: 200), () {
          _sendFrameToServer(image);
        });
      });
    });
  }
}

  Future<void> _sendFrameToServer(CameraImage image) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse('http://10.0.2.2:8000/detect/'));
      request.files.add(http.MultipartFile.fromBytes('y_plane', image.planes[0].bytes, filename: 'y_plane'));
      request.files.add(http.MultipartFile.fromBytes('u_plane', image.planes[1].bytes, filename: 'u_plane'));
      request.files.add(http.MultipartFile.fromBytes('v_plane', image.planes[2].bytes, filename: 'v_plane'));
      request.fields['width'] = image.width.toString();
      request.fields['height'] = image.height.toString();

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final jsonResponse = jsonDecode(responseBody);

        setState(() {
          leftEyeStatus = jsonResponse[0]['left_eye_status'];
          rightEyeStatus = jsonResponse[0]['right_eye_status'];
          mar = jsonResponse[0]['mar'];
          yawnDetected = jsonResponse[0]['yawn_detected'];
          somnolenciaPuntuacion = jsonResponse[0]['somnolencia_puntuacion'];
          totalBlinks = jsonResponse[0]['total_blinks'];
          totalYawns = jsonResponse[0]['total_yawns'];

          microsuenosAcumulados = jsonResponse[0]['microsuenos_acumulados'];
          blinkRate60s = jsonResponse[0]['blink_rate_60s'];
          yawnRate60s = jsonResponse[0]['yawn_rate_60s'];
        });

        _checkForAlert();  
      } else {
        print('Failed to send frame. Status code: ${response.statusCode}');
      }
    } catch (e) {
      print('Error sending frame: $e');
    }
  }

  void _checkForAlert() {
    if (somnolenciaPuntuacion >= 70 && !isAlertActive && !hasAlertTriggered) {
      isAlertActive = true;
      hasAlertTriggered = true;
      _showAlert();
    } else if (somnolenciaPuntuacion < 70) {
      hasAlertTriggered = false;
    }
  }

  void _showAlert() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Center(child: Text('¡Somnolencia Crítica!')),
          content: Text('Tu nivel de somnolencia es crítico. Detente y descansa.'),
          actions: [
            Center(
              child: ElevatedButton(
                child: Text('Estoy Despierto'),
                onPressed: () {
                  audioPlayer?.stop();
                  isAlertActive = false;
                  Navigator.of(context).pop();
                },
              ),
            ),
          ],
        );
      },
    );

    _playAlarm();  
  }

  void _playAlarm() {
    audioPlayer?.setReleaseMode(ReleaseMode.loop);

    audioPlayer?.play(AssetSource('alarma.mp3'));

    Future.delayed(Duration(seconds: 30), () {
      if (isAlertActive) {
        audioPlayer?.stop();
        isAlertActive = false;
      }
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    _timer?.cancel();
    audioPlayer?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Monitoreo del Conductor en Tiempo Real')),
      body: Column(
        children: [
          Expanded(
            flex: 2,
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return AspectRatio(
                    aspectRatio: _controller!.value.aspectRatio,
                    child: CameraPreview(_controller!),
                  );
                } else {
                  return Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          Expanded(
            flex: 1,
            child: Container(
              color: Colors.white,
              padding: EdgeInsets.all(10),
              child: _buildStatusDisplay(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(20),
            child: ElevatedButton(
              onPressed: _startOrStopSendingFrames,
              child: Text(isSendingFrames ? 'Finalizar viaje' : 'Iniciar viaje'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusDisplay() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Estado del Ojo Izquierdo: $leftEyeStatus'),
        Text('Estado del Ojo Derecho: $rightEyeStatus'),
        Text('MAR: $mar'),
        Text('Bostezo Detectado: $yawnDetected'),
        Text('Puntuación de Somnolencia: $somnolenciaPuntuacion'),
        Text('Total de Parpadeos: $totalBlinks'),
        Text('Total de Bostezos: $totalYawns'),
        Text('Microsueños Acumulados: $microsuenosAcumulados'),
        Text('Tasa de Parpadeos (60s): $blinkRate60s'),
        Text('Tasa de Bostezos (60s): $yawnRate60s'),
      ],
    );
  }
}
