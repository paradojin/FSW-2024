import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'dart:io';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Inicializar las cámaras disponibles en el dispositivo
  final cameras = await availableCameras();
  
  // Seleccionar la cámara frontal en lugar de la primera cámara disponible
  final frontCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.front,
  );

  runApp(MyApp(camera: frontCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  MyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Real-Time Driver Monitoring',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: CameraScreen(camera: camera),
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
  Timer? _timer;
  bool isSendingFrames = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
    );

    _initializeControllerFuture = _controller!.initialize();
    await _initializeControllerFuture;
  }

  void _startOrStopSendingFrames() {
    setState(() {
      isSendingFrames = !isSendingFrames;
      if (isSendingFrames) {
        _timer = Timer.periodic(Duration(seconds: 1), (Timer t) => _captureImage());
      } else {
        _timer?.cancel();
      }
    });
  }

  Future<void> _captureImage() async {
    try {
      final image = await _controller!.takePicture();
      await _sendFrameToServer(image);
    } catch (e) {
      print('Error capturing image: $e');
    }
  }

  Future<void> _sendFrameToServer(XFile file) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse('http://10.10.18.113:5000/detect'));  // Cambia la IP según la dirección del servidor
      request.files.add(await http.MultipartFile.fromPath('image', file.path));
      var response = await request.send();

      if (response.statusCode == 200) {
        print('Frame sent successfully');
      } else {
        print('Failed to send frame. Status code: ${response.statusCode}');
        response.stream.transform(utf8.decoder).listen((value) {
          print(value);
        });
      }
    } catch (e) {
      print('Error sending frame: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Monitoreo del conductor en tiempo real')),
      body: Stack(
        children: [
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return CameraPreview(_controller!);
              } else {
                return Center(child: CircularProgressIndicator());
              }
            },
          ),
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: ElevatedButton(
              onPressed: _startOrStopSendingFrames,
              child: Text(isSendingFrames ? 'Finalizar viaje' : 'Iniciar viaje'),
            ),
          ),
        ],
      ),
    );
  }
}
