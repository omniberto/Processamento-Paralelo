# 📌 Processamento Paralelo - DEC107

Repositório destinado à implementação das atividades da disciplina **Processamento Paralelo (DEC107)**, com foco na aplicação de técnicas de paralelismo utilizando OpenMP em algoritmos de processamento de imagens.

## 👥 Integrantes
- David Júnio Mariano dos Santos  
- Emanuel Humberto Menezes Cerqueira  

---

## 📥 Clonando o repositório

```bash
git clone <https://github.com/omniberto/Processamento-Paralelo.git>
```

```bash
cd <Processamento-Paralelo>
```

---

## ⚙️ Compilação

Antes de tudo, compile o módulo do stb:

```bash
gcc -c stb_impl.c -o stb_impl.o
```

### 🔹 Filtro Gaussiano

**Versão paralela:**
```bash
gcc -fopenmp filtro_gaussiano_parallel_rgb.c stb_impl.o -o mainparallel
```

**Versão sequencial:**
```bash
gcc -fopenmp filtro_gaussiano_serial_rgb.c stb_impl.o -o mainserial
```

### 🔹 Testes de corretude

**Sequencial:**
```bash
gcc -fopenmp ./corretude/corretude_serial_rgb.c stb_impl.o -o corretudeserial
```

**Paralelo:**
```bash
gcc -fopenmp ./corretude/corretude_parallel_rgb.c stb_impl.o -o corretudeparallel
```

---

## ▶️ Execução

```bash
./mainparallel
./mainserial
./corretudeserial
./corretudeparallel
```

---

## 🧠 Observações

- O projeto utiliza **OpenMP** para paralelização.
- O filtro Gaussiano é aplicado separadamente nos canais **RGB**.
- Foram implementadas versões **sequencial e paralela** para análise de desempenho.
- Os testes de corretude garantem que a versão paralela produz resultados adequados
