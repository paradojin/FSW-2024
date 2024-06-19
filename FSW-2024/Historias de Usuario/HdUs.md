# Criterios de Aceptación

## HDU 1: Detección de Pestañeos
**Categoría:** Importante  
**Puntos de Historia:** 3  

**Descripción:**  
Como conductor de un vehículo, quiero que el sistema de monitoreo diferencie entre parpadeos normales y pestañeos que indican somnolencia, para que el sistema pueda identificar signos de sueño basados en un umbral predefinido de tiempo de cierre de los ojos.

**Criterios de Aceptación:**  
- El sistema debe ser capaz de detectar y contar pestañeos del conductor con una precisión del 60%.
- Se considera un micro-sueño cuando se tarda más de 400 milisegundos en completar el pestañeo.
- La detección debe diferenciar entre parpadeos normales y pestañeos largos (micro-sueños).
- La información de los pestañeos debe ser almacenada y estar disponible para el sistema de alerta.

---

## HDU 2: Detección de Bostezos
**Categoría:** Importante  
**Puntos de Historia:** 5  

**Descripción:**  
Como conductor de un vehículo, quiero que el sistema de monitoreo detecte bostezos y los diferencie de movimientos bucales no relacionados con la fatiga, para que el sistema pueda identificar signos de sueño basados en mis bostezos.

**Criterios de Aceptación:**  
- El sistema debe ser capaz de detectar y contar bostezos del conductor con una precisión del 60%.
- Se considerará un bostezo cuando la apertura de la boca supere los 3 cm y la duración de este sea más de 7 segundos.
- La detección de bostezos se considerará efectiva cuando se registren la cantidad de bostezos por unidad de tiempo.

---

## HDU 3: Detección de Cabeceos
**Categoría:** Importante  

**Descripción:**  
Como conductor de un vehículo, quiero que el sistema de monitoreo detecte los cabeceos indicativos de somnolencia, para que el sistema pueda identificar signos de sueño basados en mis cabeceos.

**Criterios de Aceptación:**  
- El sistema debe ser capaz de detectar y contar los cabeceos del conductor con una precisión del 60%.
- La detección de cabeceos se considerará efectiva cuando se registren la cantidad de cabeceos por unidad de tiempo.
- Se considera un micro-sueño cuando la posición supera umbral del ángulo de la cabeza.
- La información de los cabeceos debe ser almacenada y estar disponible para el sistema de alerta.

---

## HDU 4: Alertas de Sueño al Volante
**Categoría:** Importante  
**Puntos de Historia:** 5  

**Descripción:**  
Como conductor de un vehículo, quiero que el sistema de alerta me notifique cuando se detecten signos de sueño, para poder tomar medidas preventivas y evitar accidentes.

**Criterios de Aceptación:**  
- El sistema debe ser capaz de predecir la somnolencia utilizando un modelo de aprendizaje automático que analiza el número de parpadeos, bostezos y cabeceos con una precisión del 60%.
- Cuando se detectan signos de somnolencia, la aplicación debe emitir una alarma sonora y visual.
- El sistema debe registrar datos de todas las alertas emitidas.

---

## HDU 5: Monitoreo en Segundo Plano
**Categoría:** Esencial  

**Descripción:**  
Como conductor, quiero que la aplicación funcione monitoreando mi fatiga incluso cuando esté en segundo plano para que pueda utilizar otras aplicaciones sin interrupciones.

**Criterios de Aceptación:**  
- La aplicación debe poder monitorear la fatiga del conductor mientras se ejecuta en segundo plano.
- El monitoreo en segundo plano no debe interferir con el uso de otras aplicaciones.
- La precisión del monitoreo en segundo plano debe ser comparable a la precisión cuando la aplicación está en primer plano.

---

## HDU 6: Calibración mediante Formulario
**Categoría:** Esencial  

**Descripción:**  
Como conductor, quiero calibrar la aplicación con mis datos mediante un formulario para que la detección de fatiga sea precisa y adaptada a mis características para evitar falsas alarmas.

**Criterios de Aceptación:**  
- La aplicación debe proporcionar un formulario para que el conductor ingrese sus datos biométricos.
- Los datos ingresados en el formulario deben ser utilizados para calibrar la detección de fatiga.
- La calibración debe reducir la incidencia de falsas alarmas en al menos un 20%.

---

## HDU 7: Calibración mediante Reconocimiento Facial
**Categoría:** Importante  

**Descripción:**  
Como conductor, quiero calibrar mis datos biométricos mediante un reconocimiento facial para que la detección de fatiga sea precisa y adaptada a mis características para evitar falsas alarmas.

**Criterios de Aceptación:**  
- La aplicación debe utilizar el reconocimiento facial para calibrar los datos biométricos del conductor.
- La calibración mediante reconocimiento facial debe mejorar la precisión de la detección de fatiga en al menos un 20%.
- El proceso de calibración debe ser sencillo y rápido, tomando menos de 1 minuto.

---

## HDU 8: Almacenamiento de Datos de Somnolencia
**Categoría:** Importante  

**Descripción:**  
Como conductor, quiero almacenar los datos de niveles de somnolencia para llevar un registro periódico de los niveles.

**Criterios de Aceptación:**  
- La aplicación debe registrar y almacenar datos de los niveles de somnolencia del conductor.
- Los datos almacenados deben ser accesibles para el conductor en un formato comprensible.
- La aplicación debe permitir la visualización de los datos históricos de somnolencia.

---

## HDU 9: Recolección de Datos Anónimos
**Categoría:** Esencial  

**Descripción:**  
Como coordinador de seguridad vial, quiero que la aplicación recolecte datos anónimos sobre la conducción para contribuir a la investigación y desarrollo de mejorar la seguridad vial mediante el análisis de datos.

**Criterios de Aceptación:**  
- La aplicación debe recolectar datos anónimos sobre los patrones de conducción y niveles de somnolencia.
- Los datos recolectados deben ser enviados a un servidor central para análisis.
- La recolección de datos debe cumplir con las normativas de privacidad y protección de datos.

---

## HDU 10: Información del Tiempo de Manejo
**Categoría:** Esencial  

**Descripción:**  
Como conductor, quiero recibir información del tiempo que llevo manejando, para que pueda tomar descansos adecuados y prevenir la fatiga.

**Criterios de Aceptación:**  
- La aplicación debe registrar el tiempo de manejo continuo del conductor.
- La aplicación debe notificar al conductor cuando se haya alcanzado un tiempo de manejo continuo predeterminado.
- La notificación debe ser clara y no distractora para el conductor.

---

## HDU 11: Personalización de Alertas
**Categoría:** Deseable  

**Descripción:**  
Como conductor, quiero personalizar el tipo de alertas que recibo (sonoras, vibraciones) para que las notificaciones personales a mi gusto sean efectivas y no me distraigan durante la conducción.

**Criterios de Aceptación:**  
- La aplicación debe permitir al conductor elegir entre diferentes tipos de alertas (sonoras, vibraciones).
- El conductor debe poder ajustar la intensidad y frecuencia de las alertas.
- Las alertas personalizadas deben ser efectivas y no distractoras.

---

## HDU 12: Modo de Ahorro de Batería
**Categoría:** Opcional  

**Descripción:**  
Como conductor, quiero que la aplicación tenga un modo de ahorro de batería para que no se agote rápidamente durante trayectos largos.

**Criterios de Aceptación:**  
- La aplicación debe tener un modo de ahorro de batería que reduzca el consumo de energía.
- El modo de ahorro de batería no debe comprometer significativamente la precisión del monitoreo de fatiga.
- El conductor debe poder activar y desactivar el modo de ahorro de batería fácilmente.

---

## HDU 13: Ajuste Automático de Brillo y Colores
**Categoría:** Opcional  

**Descripción:**  
Como conductor, quiero que la aplicación ajuste automáticamente el brillo y los colores de la interfaz para la conducción nocturna para no ser deslumbrado y mantener la visibilidad.

**Criterios de Aceptación:**  
- La aplicación debe ajustar automáticamente el brillo de la pantalla en condiciones de poca luz.
- La aplicación debe ajustar los colores de la interfaz para ser más suaves y adecuados para la conducción nocturna.
- Los ajustes automáticos deben poder ser personalizados por el conductor.
