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

__global__ void kernelFase03_Simple(int* d_data, int num_vuelos, int* d_res, bool buscarMaximo) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < num_vuelos) {

        int llave = d_data[id];

        if (buscarMaximo) {
            if (llave != INT_MIN) {
                atomicMax(d_res, llave);

            }
            else {
                if (llave != INT_MAX) {
                    atomicMin(d_res, llave);
                }
            }
        }

    }


}

__global__ void kernelFase03_Basica(int* d_data, int num_vuelos, int* d_res, bool buscarMaximo) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int NULO;
    if (buscarMaximo)
        NULO = INT_MIN;
    else
        NULO = INT_MAX;

    if (idx < num_vuelos) {
        shared_data[tid + 1] = d_data[idx];
    }
    else {
        shared_data[tid + 1] = NULO;
    }
    if (tid == 0) {
        if (idx > 0) {
            shared_data[0] = d_data[idx - 1];
        }
        else {
            shared_data[0] = NULO;
        }
    }

    if (tid == blockDim.x - 1) {
        if (idx < num_vuelos - 1) {
            shared_data[blockDim.x + 1] = d_data[idx + 1];
        }
        else {
            shared_data[blockDim.x + 1] = NULO;
        }
    }

    __syncthreads();

    // Guarda: solo actuar si el dato propio es valido
    if (idx < num_vuelos && shared_data[tid + 1] != NULO)
    {
        int anterior = shared_data[tid];
        int actual = shared_data[tid + 1];
        int posterior = shared_data[tid + 2];
        int mejor = actual;

        if (buscarMaximo) {
            if (anterior != NULO && anterior > mejor) {
                mejor = anterior;
            }
            if (posterior != NULO && posterior > mejor) {
                mejor = posterior;
            }
            if (mejor != NULO) {
                atomicMax(d_res, mejor);
            }
        }
        else {
            if (anterior != NULO && anterior < mejor) {
                mejor = anterior;
            }
            if (posterior != NULO && posterior < mejor) {
                mejor = posterior;
            }
            if (mejor != NULO) {
                atomicMin(d_res, mejor);
            }
        }
    }
}

// Kernel Fase 03 - Variante 3.3: Intermedia (Ventana de 3 + Reduccion de pares)
__global__ void kernelFase03_Intermedia(int* d_data, int num_vuelos, int* d_res, bool buscarMaximo) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int NULO;
    if (buscarMaximo)
        NULO = INT_MIN;
    else
        NULO = INT_MAX;

    // --- FASE 1: Carga de datos y halos en shared_data ---
    if (idx < num_vuelos) {
        shared_data[tid + 1] = d_data[idx];
    }
    else {
        shared_data[tid + 1] = NULO;
    }

    if (tid == 0) {
        if (idx > 0) {
            shared_data[0] = d_data[idx - 1];
        }
        else {
            shared_data[0] = NULO;
        }
    }

    if (tid == blockDim.x - 1) {
        if (idx < num_vuelos - 1) {
            shared_data[blockDim.x + 1] = d_data[idx + 1];
        }
        else {
            shared_data[blockDim.x + 1] = NULO;
        }
    }

    __syncthreads();

    // --- FASE 2: Calcular el mejor de la ventana y guardarlo localmente ---
    int local_mejor = NULO;

    // Guarda: solo actuar si el dato propio es valido
    if (idx < num_vuelos && shared_data[tid + 1] != NULO)
    {
        int anterior = shared_data[tid];
        int actual = shared_data[tid + 1];
        int posterior = shared_data[tid + 2];
        local_mejor = actual;

        if (buscarMaximo) {
            if (anterior != NULO && anterior > local_mejor) {
                local_mejor = anterior;
            }
            if (posterior != NULO && posterior > local_mejor) {
                local_mejor = posterior;
            }
        }
        else {
            if (anterior != NULO && anterior < local_mejor) {
                local_mejor = anterior;
            }
            if (posterior != NULO && posterior < local_mejor) {
                local_mejor = posterior;
            }
        }
    }

    __syncthreads();

    shared_data[tid + 1] = local_mejor;

    __syncthreads();

    if (idx < num_vuelos && tid % 2 == 0) {
        int mi_valor = shared_data[tid + 1];

        int valor_siguiente = NULO;
        if (idx + 1 < num_vuelos && (tid + 1) < blockDim.x) {
            valor_siguiente = shared_data[tid + 2];
        }

        int mejor_final = mi_valor;

        if (buscarMaximo) {
            if (valor_siguiente != NULO && valor_siguiente > mejor_final) {
                mejor_final = valor_siguiente;
            }
            if (mejor_final != NULO) {
                atomicMax(d_res, mejor_final);
            }
        }
        else {
            if (valor_siguiente != NULO && valor_siguiente < mejor_final) {
                mejor_final = valor_siguiente;
            }
            if (mejor_final != NULO) {
                atomicMin(d_res, mejor_final);
            }
        }
    }
}

// Kernel Fase 03 - Variante 3.4: Patron de Reduccion Optimizada
__global__ void kernelFase03_Reduccion(int* d_in, int n, int* d_out, bool buscarMaximo) {
    extern __shared__ int s_data[]; // Pizarra dinámica

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int NULO;
    if (buscarMaximo)
        NULO = INT_MIN;
    else
        NULO = INT_MAX;

    if (idx < n)
        s_data[tid] = d_in[idx];
    else
        s_data[tid] = NULO;


    __syncthreads();

    // 2. Reducción en Árbol Optimizada (Evitando divergencia de warps)
    // El tamaño del salto 's' se divide a la mitad en cada ronda
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int mi_valor = s_data[tid];
            int rival = s_data[tid + s];

            if (buscarMaximo) {
                if (rival != NULO && rival > mi_valor) {
                    s_data[tid] = rival;
                }
            }
            else {
                if (rival != NULO && rival < mi_valor) {
                    s_data[tid] = rival;
                }
            }
        }
        __syncthreads(); // Esperamos a que termine esta ronda del torneo
    }

    // 3. El Hilo 0 de cada bloque anota al Campeón del bloque en la memoria global
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
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

            cout << "\nNumero de vuelos: " << num_vuelos << endl;

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
            if (hilosPorBloque > 256) {
                hilosPorBloque = 256;
            }

            int bloquesPorGrid =
                (num_vuelos + hilosPorBloque - 1) / hilosPorBloque;

            cout << "\nConfiguracion GPU:\n";
            cout << "Bloques: " << bloquesPorGrid << endl;
            cout << "Hilos por bloque: " << hilosPorBloque << endl;

            // ============================================
            // ejecutar kernel
            // ============================================

            kernelFase02 << <bloquesPorGrid, hilosPorBloque >> > (
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
        {
            cout << "\n--- Iniciando Fase 03 (Variante 3.1): Reduccion Simple ---\n";

            int columna_opcion;
            do {
                cout << "Seleccione la columna:\n (1) DEP_DELAY\n (2) ARR_DELAY\n (3) WEATHER_DELAY\n Opcion: ";
                cin >> columna_opcion;

                if (columna_opcion < 1 || columna_opcion > 3)
                    cout << "Opcion no valida, intentelo de nuevo.\n";

            } while (columna_opcion < 1 || columna_opcion > 3);

            // 2. Menú de Operación
            int tipo_opcion;
            do {
                cout << "Busqueda:\n (1) Maximo\n (2) Minimo\n Opcion: ";
                cin >> tipo_opcion;

                if (tipo_opcion != 1 && tipo_opcion != 2)
                    cout << "Opcion no valida, intentelo de nuevo.\n";

            } while (tipo_opcion != 1 && tipo_opcion != 2);

            bool buscarMaximo;
            if (tipo_opcion == 1)
                buscarMaximo = true;
            else
                buscarMaximo = false;

            // 3. Selección del origen de datos
            vector<float>* columna_origen;
            if (columna_opcion == 1) {
                columna_origen = &dataset.dep_delay;
            }
            else if (columna_opcion == 2) {
                columna_origen = &dataset.arr_delay;
            }
            else {
                columna_origen = &dataset.weather_delay;
            }

            int variante;
            do {
                cout << "Seleccione Variante (1=Simple, 2=Basica, 3=Intermedio, 4=Reduccion): ";
                cin >> variante;

                if (variante < 1 || variante > 4) {
                    cout << "Opcion no valida, intentelo de nuevo.\n";
                }

            } while (variante < 1 || variante > 4);

            // 4. Truncado de float a int (Exigencia del enunciado)
            int num_vuelos = (*columna_origen).size();
            vector<int> datos_int(num_vuelos);

            // Usamos un valor trampa para los datos corruptos
            int nulo_val = buscarMaximo ? INT_MIN : INT_MAX;

            for (int i = 0; i < num_vuelos; i++) {
                float val_f = (*columna_origen)[i];
                if (isnan(val_f)) {
                    datos_int[i] = nulo_val; // Dato corrupto, se ignora luego
                }
                else {
                    datos_int[i] = (int)val_f; // Casteo: Trunca los decimales automáticamente
                }
            }

            // 5. Comprobación de Hardware 
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, 0);
            int hilosPorBloque = props.maxThreadsPerBlock;
            if (hilosPorBloque > 256) {
                hilosPorBloque = 256;
            }
            int bloquesPorGrid = (num_vuelos + hilosPorBloque - 1) / hilosPorBloque;
            cout << "\n[Hardware] Tarjeta: " << props.name << endl;


            // 6. Preparar Memoria VRAM
            size_t bytes_datos = num_vuelos * sizeof(int);
            int* d_data;
            int* d_res;

            cudaMalloc(&d_data, bytes_datos);
            cudaMalloc(&d_res, sizeof(int));

            cudaMemcpy(d_data, datos_int.data(), bytes_datos, cudaMemcpyHostToDevice);
            cudaMemcpy(d_res, &nulo_val, sizeof(int), cudaMemcpyHostToDevice); // Inicializamos el resultado al peor caso

            // 8. Lanzamiento de Kernels
            cout << "Procesando Variante " << variante << "...\n";

            if (variante == 1) {
                // Variante 3.1: Lanzamiento normal sin memoria compartida extra
                kernelFase03_Simple << <bloquesPorGrid, hilosPorBloque >> > (d_data, num_vuelos, d_res, buscarMaximo);
            }
            else if (variante == 2) {
                // Variante 3.2: Calculamos los BYTES de Memoria Compartida
                // Formula = (Cantidad de Hilos + 2 Huecos para Halos) * Tamaño de un Entero
                size_t sharedMemBytes = (hilosPorBloque + 2) * sizeof(int);

                cout << "[Memoria] Asignando " << sharedMemBytes << " bytes de Shared Memory por bloque.\n";

                // ¡Fíjate en el TERCER parámetro dentro de los corchetes!
                kernelFase03_Basica << <bloquesPorGrid, hilosPorBloque, sharedMemBytes >> > (d_data, num_vuelos, d_res, buscarMaximo);
            }
            else if (variante == 3) {
                // Variante 3.3: Misma reserva de memoria porque empezamos con la ventana de 3
                size_t sharedMemBytes = (hilosPorBloque + 2) * sizeof(int);
                cout << "[Variante 3.3] Lanzando reduccion por pares con " << sharedMemBytes << " bytes/bloque.\n";
                kernelFase03_Intermedia << <bloquesPorGrid, hilosPorBloque, sharedMemBytes >> > (d_data, num_vuelos, d_res, buscarMaximo);
            }
            else if (variante == 4) {
                // Variante 3.4: Reducciones Sucesivas
                int N_actual = num_vuelos;
                int N_bloques = bloquesPorGrid;

                // Memoria para el torneo
                int* d_in_torneo;
                int* d_out_torneo;
                cudaMalloc(&d_in_torneo, N_actual * sizeof(int));
                cudaMalloc(&d_out_torneo, N_bloques * sizeof(int));

                // Copiamos los datos originales a la arena de entrada
                cudaMemcpy(d_in_torneo, d_data, N_actual * sizeof(int), cudaMemcpyDeviceToDevice);

                int rondas = 1;
                while (true) {
                    size_t sharedMemBytes = hilosPorBloque * sizeof(int); // Ahora no hay halos
                    cout << "[Ronda " << rondas << "] Entran " << N_actual << " vuelos. Se generan " << N_bloques << " bloques.\n";

                    kernelFase03_Reduccion << <N_bloques, hilosPorBloque, sharedMemBytes >> > (d_in_torneo, N_actual, d_out_torneo, buscarMaximo);
                    cudaDeviceSynchronize();

                    // Si ya quedan 10 campeones o menos, terminamos el bucle de la GPU
                    if (N_bloques <= 10)
                    {
                        break;
                    }

                    // Si quedan más de 10, preparamos la SIGUIENTE RONDA
                    N_actual = N_bloques;
                    N_bloques = (N_actual + hilosPorBloque - 1) / hilosPorBloque;

                    // Los campeones de hoy (d_out) son los contendientes de mañana (d_in)
                    cudaMemcpy(d_in_torneo, d_out_torneo, N_actual * sizeof(int), cudaMemcpyDeviceToDevice);
                    rondas++;
                }

                // --- POST-PROCESADO EN CPU ---
                // Rescatamos los campeones finales (máximo 10)
                vector<int> campeones_finales(N_bloques);
                cudaMemcpy(campeones_finales.data(), d_out_torneo, N_bloques * sizeof(int), cudaMemcpyDeviceToHost);

                cout << "\n[CPU] Rescatados " << N_bloques << " resultados finales. Resolviendo en CPU...\n";

                // Iteramos en la CPU para el ganador absoluto
                int ganador_absoluto = campeones_finales[0];
                for (int i = 1; i < N_bloques; i++) {
                    if (buscarMaximo) {
                        if (campeones_finales[i] > ganador_absoluto) ganador_absoluto = campeones_finales[i];
                    }
                    else {
                        if (campeones_finales[i] < ganador_absoluto) ganador_absoluto = campeones_finales[i];
                    }
                }

                // Sobrescribimos la variable de resultado para que se imprima al final
                cudaMemcpy(d_res, &ganador_absoluto, sizeof(int), cudaMemcpyHostToDevice);

                cudaFree(d_in_torneo);
                cudaFree(d_out_torneo);
            }
            else {
                cout << "\n[Aviso] Variante no implementada aun.\n";
            }
            cudaDeviceSynchronize();

            // 8. Traer resultado y mostrar
            int resultado_final;
            cudaMemcpy(&resultado_final, d_res, sizeof(int), cudaMemcpyDeviceToHost);

            cout << "\n=======================================\n";
            cout << ">>> EL RESULTADO " << (buscarMaximo ? "MAXIMO" : "MINIMO") << " ES: " << resultado_final << " minutos <<<";
            cout << "\n=======================================\n";

            // 9. Limpiar
            cudaFree(d_data);
            cudaFree(d_res);
            break;
        }
        case '4':
        {
            cout << "\n--- Iniciando Fase 04: Histograma de aeropuertos ---\n";
            // TODO: Mostrar submenú y lanzar Kernel de la Fase 4
            break;
        }

        case 'x':
            cout << "\nSaliendo del programa...\n";
            break;
        default:
            cout << "\nOpcion no valida. Intente de nuevo.\n";
        }
    } while (opcion != 'x');

    return 0;
}
