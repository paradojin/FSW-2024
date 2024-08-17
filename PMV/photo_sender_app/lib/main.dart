import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'package:http/http.dart' as http;
import 'dart:io';

List<CameraDescription> cameras = [];
CameraController? controller;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  // Selecciona la cámara frontal
  controller = CameraController(cameras.firstWhere((camera) => camera.lensDirection == CameraLensDirection.front), ResolutionPreset.medium);
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    controller?.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
    });
    _timer = Timer.periodic(Duration(seconds: 2), (Timer t) => _captureAndSendImage());
  }

  Future<void> _captureAndSendImage() async {
    try {
      final image = await controller?.takePicture();
      if (image != null) {
        final file = File(image.path);
        final uri = Uri.parse('http://192.168.1.83:5000/detect');  // Cambia a la IP de tu PC
        final request = http.MultipartRequest('POST', uri)
          ..files.add(await http.MultipartFile.fromPath('image', file.path));
        final response = await request.send();
        if (response.statusCode == 200) {
          print('Image sent successfully');
        } else {
          print('Failed to send image');
        }
      }
    } catch (e) {
      print('Error capturing or sending image: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Camera Example'),
      ),
      body: controller?.value.isInitialized ?? false
          ? CameraPreview(controller!)
          : Center(child: CircularProgressIndicator()),
    );
  }

  @override
  void dispose() {
    _timer.cancel();
    controller?.dispose();
    super.dispose();
  }
}
