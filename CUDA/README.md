# 📌 Processamento Paralelo - DEC107

Repositório destinado à implementação das atividades da disciplina **Processamento Paralelo (DEC107)**, com foco na aplicação de técnicas de paralelismo utilizando CUDA em algoritmos de processamento de imagens.

## 👥 Integrantes
- David Júnio Mariano dos Santos  
- Emanuel Humberto Menezes Cerqueira  

---

## 📥 Clonando o repositório

```bash
git clone https://github.com/omniberto/Processamento-Paralelo.git
```

```bash
cd Processamento-Paralelo
```
---
## 📁 Acessando a Pasta
```bash
cd CUDA
```

---

## ⚙️ Compilação

Antes de tudo, compile o módulo do **stb**. Os comandos variam de acordo com o sistema operacional.

### 🪟 Windows

No Windows, é necessário gerar tanto o arquivo **`.o`** (utilizado pelo **GCC** para a versão serial e os testes de corretude) quanto o arquivo **`.obj`** (utilizado pelo **NVCC** durante a compilação do código CUDA via **MSVC**).

#### Gerar `stb_impl.o`
```bash
gcc -c stb_impl.c -o stb_impl.o
```

#### Gerar `stb_impl.obj`
```cmd
cl /c stb_impl.c /Fostb_impl.obj
```

---

### 🐧 Linux

No Linux, o arquivo **`.obj`** não é necessário. O **NVCC** utiliza o **GCC/G++** como compilador host e aceita nativamente arquivos **`.o`**.

#### Gerar `stb_impl.o`
```bash
gcc -c stb_impl.c -o stb_impl.o
```

### 🔹 Filtro Gaussiano

#### 🪟 Windows

**Versão com memória global:**
```cmd
nvcc -Xcompiler /openmp filtro_gaussiano_cuda_rgb.cu stb_impl.obj -o maincuda
```

**Versão com shared memory:**
```cmd
nvcc -Xcompiler /openmp filtro_gaussiano_cuda_shared_rgb.cu stb_impl.obj -o maincuda_shared
```

**Versão sequencial:**
```cmd
gcc -fopenmp filtro_gaussiano_serial_rgb.c stb_impl.o -o mainserial
```

---

#### 🐧 Linux

**Versão com memória global:**
```bash
nvcc -Xcompiler -fopenmp filtro_gaussiano_cuda_rgb.cu stb_impl.o -o maincuda
```

**Versão com shared memory:**
```bash
nvcc -Xcompiler -fopenmp filtro_gaussiano_cuda_shared_rgb.cu stb_impl.o -o maincuda_shared
```

**Versão sequencial:**
```bash
gcc -fopenmp filtro_gaussiano_serial_rgb.c stb_impl.o -o mainserial
```

---

### Corretude

**Versão com memória global:**
```bash
gcc -fopenmp ./corretude/corretude.c stb_impl.o -o corretude
```

**Versão com shared memory:**
```bash
gcc -fopenmp ./corretude/corretude_shared.c stb_impl.o -o corretudeshared
```
## ▶️ Execução

```bash
./maincuda
```
```bash
./maincuda_shared
```
```bash
./mainserial
```
```bash
./corretude
```
```bash
./corretudeshared
```

---

## 🧠 Observações

- O projeto utiliza **CUDA** para paralelização massiva em GPU.
- O filtro Gaussiano é aplicado separadamente nos canais **RGB**.
- Foram implementadas duas versões CUDA: uma com acesso direto à **memória global** e outra com uso de **shared memory**.
- Os testes de corretude garantem que ambas as versões CUDA produzem resultados idênticos à versão serial.
- Os experimentos foram realizados com a GPU **NVIDIA GeForce RTX 4050 Laptop GPU** (arquitetura Ada Lovelace, compute capability 8.9).

## Nota de Transparência sobre o Uso de IA:
Declaro que este projeto contou com o auxílio das ferramentas de IA exclusivamente para as tarefas de

- Revisão gramatical e tradução de trechos do relatório
- Auxílio na depuração (debugging) de blocos específicos de código
- Geração de scripts base para automação de testes de desempenho
- Estruturação inicial de ideias e revisão bibliográfica
- Estruturação de texto em formato LaTeX

Como autor(a), atesto que revisei, testei e validei criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos relatórios de desempenho e integridade acadêmica do material entregue.

[Emanuel Humberto Menezes Cerqueira] – [04 de julho de 2026]

[David Júnio Mariano dos Santos] – [04 de julho de 2026]
