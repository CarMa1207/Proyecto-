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
  
    int id = blockIdx.x * blockDim.x + threadIdx.x;

 
    if (id < num_vuelos) {
        float retraso_vuelo = d_retrasos_despegue[id];

        //  Ignoramos los valores nulos/vacíos (NAN)
        if (!isnan(retraso_vuelo)) {
        
            if (umbral_usuario >= 0 && retraso_vuelo >= umbral_usuario) {
                printf("- Hilo #%d: Retraso de %.0f minutos\n", id, retraso_vuelo);
            }
            else if (umbral_usuario < 0 && retraso_vuelo <= umbral_usuario) {
                printf("Hilo #%d: Adelanto de %.0f minutos\n", id, retraso_vuelo);
            }
        }
    }
}

// FASE 02
// memoria constante para el umbral
__constant__ int d_umbral;


// Kernel CUDA Fase 02
__global__ void kernelFase02(
    float* d_retraso_despegue,
    char* d_tail_nums,
    int num_vuelos,
    int* d_contador,
    float* d_result_retrasos,
    char* d_result_tail
)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < num_vuelos)
    {
        float retraso = d_retraso_despegue[id];

        if (!isnan(retraso))
        {
            bool cumple = false;

            // caso retraso
            if (d_umbral >= 0 && retraso >= d_umbral)
                cumple = true;

            // caso adelanto
            if (d_umbral < 0 && retraso <= d_umbral)
                cumple = true;

            if (cumple)
            {
                // posicion segura en array resultado
                int pos = atomicAdd(d_contador, 1);

                d_result_retrasos[pos] = retraso;

                // copiar matrícula (8 chars máximo)
                for (int j = 0; j < 8; j++)
                {
                    d_result_tail[pos * 8 + j] =
                        d_tail_nums[id * 8 + j];
                }

                printf(
                    "- Hilo #%d | Matricula: %.8s | Retraso (llegada): %.0f min\n",
                    id,
                    &d_tail_nums[id * 8],
                    retraso
                );
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
            cout << "Introduzca el umbral en minutos (Positivo = Retraso, Negativo = Adelanto) ";
            cin >> umbral;

            int num_vuelos = dataset.dep_delay.size();
            size_t bytes = num_vuelos * sizeof(float);

            float* d_dep_delay;
            cudaMalloc(&d_dep_delay, bytes);

            cout << "Transfiriendo " << num_vuelos << " registros a la GPU..." << endl;
            cudaMemcpy(d_dep_delay, dataset.dep_delay.data(), bytes, cudaMemcpyHostToDevice);

            
            int hilosPorBloque = 256;
            int bloquesPorGrid = (num_vuelos + hilosPorBloque - 1) / hilosPorBloque;

            cout << "Lanzando Kernel: " << bloquesPorGrid << " bloques de " << hilosPorBloque << " hilos\n\n";

           
            kernelFase01 << <bloquesPorGrid, hilosPorBloque >> > (d_dep_delay, num_vuelos, umbral);

            
            cudaDeviceSynchronize();

         
            cudaFree(d_dep_delay);
            break;
        }
        case '2':
        {
            cout << "\n--- Iniciando Fase 02: Retraso en aterrizajes ---\n";

            int umbral;
            cout << "Introduzca el umbral en minutos (positivo=retraso, negativo=adelanto): ";
            cin >> umbral;

            int num_vuelos = dataset.arr_delay.size();

            cout << "\nNumero de vuelos a analizar: " << num_vuelos << endl;

            // ============================================
            // convertir vector<string> tail_num a array char linealizado
            // ============================================

            const int MAX_TAIL = 8;

            vector<char> tail_nums_lineal(num_vuelos * MAX_TAIL);

            for (int i = 0; i < num_vuelos; i++)
            {
                string matricula = dataset.tail_num[i];

                for (int j = 0; j < MAX_TAIL; j++)
                {
                    if (j < matricula.size())
                        tail_nums_lineal[i * MAX_TAIL + j] = matricula[j];
                    else
                        tail_nums_lineal[i * MAX_TAIL + j] = '\0';
                }
            }


            float* d_arr_delay;
            char* d_tail_nums;

            cudaMalloc(&d_arr_delay, num_vuelos * sizeof(float));
            cudaMalloc(&d_tail_nums, num_vuelos * MAX_TAIL);

            cudaMemcpy(
                d_arr_delay,
                dataset.arr_delay.data(),
                num_vuelos * sizeof(float),
                cudaMemcpyHostToDevice
            );

            cudaMemcpy(
                d_tail_nums,
                tail_nums_lineal.data(),
                num_vuelos * MAX_TAIL,
                cudaMemcpyHostToDevice
            );

            cudaMemcpyToSymbol(
                d_umbral,
                &umbral,
                sizeof(int)
            );

            float* d_result_delays;
            char* d_result_tail;
            int* d_contador;

            cudaMalloc(&d_result_delays, num_vuelos * sizeof(float));
            cudaMalloc(&d_result_tail, num_vuelos * MAX_TAIL);
            cudaMalloc(&d_contador, sizeof(int));

            cudaMemset(d_contador, 0, sizeof(int));

            // configuracion GPU dinamica

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            int hilosPorBloque = prop.maxThreadsPerBlock;

            int bloquesPorGrid =
                (num_vuelos + hilosPorBloque - 1) / hilosPorBloque;

            cout << "\nConfiguracion GPU:\n";
            cout << "Bloques: " << bloquesPorGrid << endl;
            cout << "Hilos por bloque: " << hilosPorBloque << endl;

            // ============================================
            // ejecutar kernel
            // ============================================

            kernelFase02 <<<bloquesPorGrid, hilosPorBloque >>> (
                d_arr_delay,
                d_tail_nums,
                num_vuelos,
                d_contador,
                d_result_delays,
                d_result_tail
                );

            cudaDeviceSynchronize();

            // ============================================
            // recuperar resultados
            // ============================================

            int total_detectados;

            cudaMemcpy(
                &total_detectados,
                d_contador,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            vector<float> h_result_delays(total_detectados);
            vector<char> h_result_tail(total_detectados * MAX_TAIL);

            cudaMemcpy(
                h_result_delays.data(),
                d_result_delays,
                total_detectados * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                h_result_tail.data(),
                d_result_tail,
                total_detectados * MAX_TAIL,
                cudaMemcpyDeviceToHost
            );

            // ============================================
            // mostrar resumen final
            // ============================================

            cout << "\nResultados encontrados: " << total_detectados << " vuelos\n";

            for (int i = 0; i < total_detectados && i < 10; i++)
            {
                cout
                    << "Matricula: "
                    << &h_result_tail[i * MAX_TAIL]
                    << " Retraso: "
                    << h_result_delays[i]
                    << " minutos\n";
            }

            cout << "\n(se muestran maximo 10 resultados)\n";

            // liberar memoria GPU

            cudaFree(d_arr_delay);
            cudaFree(d_tail_nums);
            cudaFree(d_result_delays);
            cudaFree(d_result_tail);
            cudaFree(d_contador);

            cout << "\nFase 02 completada.\n";

            break;
        }
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
