#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <string>

#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>

#include <limits.h>


using namespace std;

// Estructura para agrupar las columnas de nuestro dataset
struct AirlineDataset {
    vector<float> dep_delay;
    vector<float> arr_delay;
    vector<float> weather_delay;
    vector<string> tail_num;
    vector<string> origin_airport;
    vector<string> dest_airport;
    vector<int> origin_seq_id;
    vector<int> dest_seq_id;
    // Añadiremos más campos si las fases lo requieren
};


// Función auxiliar para convertir a float controlando los valores vacíos
float obtenerFloat(const vector<string>& fila, int indice) {
    if (indice >= fila.size() || fila[indice].empty()) {
        return NAN; // Si no hay valor, guardamos NAN como pide el enunciado
    }
    try {
        return stof(fila[indice]);
    }
    catch (...) {
        return NAN; // Por si hay algún carácter extraño
    }
}

// Función principal de lectura
bool cargarDatosCSV(const string& ruta, AirlineDataset& dataset) {
    ifstream archivo(ruta);
    if (!archivo.is_open()) {
        cerr << "Error: No se pudo abrir el archivo CSV en la ruta: " << ruta << endl;
        return false;
    }

    string linea;
    // 1. Leer la primera línea (cabeceras) para mapear dinámicamente las columnas
    if (!getline(archivo, linea)) return false;

    unordered_map<string, int> mapaColumnas;
    stringstream ss_cabecera(linea);
    string nombreCol;
    int indice = 0;
    while (getline(ss_cabecera, nombreCol, ',')) {
        if (!nombreCol.empty() && nombreCol.back() == '\r') nombreCol.pop_back(); // Limpiar retorno de carro
        mapaColumnas[nombreCol] = indice;
        indice++;
    }

    // 2. Procesar el resto del archivo línea a línea
    int lineasProcesadas = 0;
    while (getline(archivo, linea)) {
        if (linea.empty()) continue;

        vector<string> fila;
        stringstream ss_fila(linea);
        string valor;
        // Separamos los valores de la línea por comas
        while (getline(ss_fila, valor, ',')) {
            if (!valor.empty() && valor.back() == '\r') valor.pop_back();
            fila.push_back(valor);
        }

        // 3. Volcar los datos en nuestra estructura SoA
        if (mapaColumnas.count("DEP_DELAY")) dataset.dep_delay.push_back(obtenerFloat(fila, mapaColumnas["DEP_DELAY"]));
        if (mapaColumnas.count("ARR_DELAY")) dataset.arr_delay.push_back(obtenerFloat(fila, mapaColumnas["ARR_DELAY"]));
        if (mapaColumnas.count("WEATHER_DELAY")) dataset.weather_delay.push_back(obtenerFloat(fila, mapaColumnas["WEATHER_DELAY"]));

        if (mapaColumnas.count("TAIL_NUM")) dataset.tail_num.push_back(fila[mapaColumnas["TAIL_NUM"]]);
        if (mapaColumnas.count("ORIGIN_AIRPORT")) dataset.origin_airport.push_back(fila[mapaColumnas["ORIGIN_AIRPORT"]]);
        if (mapaColumnas.count("DEST_AIRPORT")) dataset.dest_airport.push_back(fila[mapaColumnas["DEST_AIRPORT"]]);

        lineasProcesadas++;
    }

    archivo.close();
    cout << "Dataset cargado con exito. Filas leidas: " << lineasProcesadas << endl;
    return true;
}

// =========================================================================
// KERNELS DE CUDA
// =========================================================================

// Kernel Fase 01: Detección de retrasos en despegues (DEP_DELAY)
__global__ void kernelFase01(float* d_retrasos_despegue, int num_vuelos, int umbral_usuario) {
    // 1. Calculamos la posición exacta (índice 1D) de este hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Comprobamos que no nos salimos del tamaño del vector
    if (idx < num_vuelos) {
        float retraso_vuelo = d_retrasos_despegue[idx];

        // 3. Ignoramos los valores nulos/vacíos (NAN)
        if (!isnan(retraso_vuelo)) {
            // Evaluamos si es un retraso (umbral positivo)
            if (umbral_usuario >= 0 && retraso_vuelo >= umbral_usuario) {
                printf("- Hilo #%d: Retraso de %.0f minutos\n", idx, retraso_vuelo);
            }
            // Evaluamos si es un adelanto (umbral negativo)
            else if (umbral_usuario < 0 && retraso_vuelo <= umbral_usuario) {
                printf("Hilo #%d: Adelanto de %.0f minutos\n", idx, retraso_vuelo);
            }
        }
    }
}



int main() {
    string rutaCSV = "C:\\Users\\carlos.martinezarias\\Documents\\Airline_dataset.csv";
    cout << "PAP 2026 - PL1 CUDA" << endl;
   /*
    
    cout << "Introduzca la ruta base del dataset (pulse Intro para usar por defecto: C:\\dataset.csv): ";

    getline(cin, rutaCSV);
    if (rutaCSV.empty()) {
        rutaCSV = "C:\\dataset.csv"; // Ruta por defecto sugerida en el enunciado
    }

    */

    cout << "\nCargando dataset desde: " << rutaCSV << " ..." << endl;
    AirlineDataset dataset;
    if (!cargarDatosCSV(rutaCSV, dataset)) {
        cout << "Fallo al cargar los datos. Saliendo del programa..." << endl;
        return 1;
    }

    char opcion;
    do {
        cout << "\nMenu de opciones:\n";
        cout << "(1) Retraso en salida.\n";
        cout << "(2) Retraso en llegada.\n";
        cout << "(3) Reduccion de retraso.\n";
        cout << "(4) Histograma de aeropuertos.\n";
        cout << "(x) Salir\n";
        cout << "Seleccione una opcion: ";
        cin >> opcion;

        switch (opcion) {
        case '1':
        {   
            cout << "\n--- Iniciando Fase 01: Retraso en despegues ---\n";
            int umbral;
            cout << "Introduzca el umbral en minutos (Positivo = Retraso, Negativo = Adelanto): ";
            cin >> umbral;

            int num_vuelos = dataset.dep_delay.size();
            size_t bytes = num_vuelos * sizeof(float);

            // 1. Reservar memoria en la tarjeta gráfica (VRAM)
            float* d_dep_delay;
            cudaMalloc(&d_dep_delay, bytes);

            // 2. Transferir los datos de la CPU a la GPU
            cout << "Transfiriendo " << num_vuelos << " registros a la GPU..." << endl;
            cudaMemcpy(d_dep_delay, dataset.dep_delay.data(), bytes, cudaMemcpyHostToDevice);

            // 3. Configurar la topología de la GPU
            int hilosPorBloque = 256;
            int bloquesPorGrid = (num_vuelos + hilosPorBloque - 1) / hilosPorBloque;

            cout << "Lanzando Kernel: " << bloquesPorGrid << " bloques de " << hilosPorBloque << " hilos...\n\n";

            // 4. Ejecución masiva en paralelo
            kernelFase01<<<bloquesPorGrid, hilosPorBloque>>>(d_dep_delay, num_vuelos, umbral);

            // 5. La CPU espera a que la gráfica termine de imprimir todo
            cudaDeviceSynchronize();

            // 6. Limpiar la memoria gráfica
            cudaFree(d_dep_delay);
            cout << "\nFase 01 completada.\n";
            break;
        }   
        case '2':
            cout << "\n--- Iniciando Fase 02: Retraso en aterrizajes ---\n";
            // TODO: Pedir umbral y lanzar Kernel de la Fase 2
            break;
        case '3':
            cout << "\n--- Iniciando Fase 03: Reduccion de retraso ---\n";
            // TODO: Mostrar submenú y lanzar Kernel de la Fase 3
            break;
        case '4':
            cout << "\n--- Iniciando Fase 04: Histograma de aeropuertos ---\n";
            // TODO: Mostrar submenú y lanzar Kernel de la Fase 4
            break;
        case 'x':
            cout << "\nSaliendo del programa...\n";
            break;
        default:
            cout << "\nOpcion no valida. Intente de nuevo.\n";
        }
    } while (opcion != 'x');

    return 0;
}
