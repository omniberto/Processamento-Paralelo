# 📌 Processamento Paralelo - DEC107

Repositório destinado à implementação das atividades da disciplina **Processamento Paralelo (DEC107)**, com foco na aplicação de técnicas de paralelismo utilizando OpenMP em algoritmos de processamento de imagens.

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
cd MPI
```
ou
```bash
cd OpenMP
```
---

## ⚙️ Compilação

Antes de tudo, compile o módulo do stb:

```bash
gcc -c stb_impl.c -o stb_impl.o
```

### 🔹 Filtro Gaussiano

**Versão usando MPI:**
#### SEM mpicc
```bash
gcc -fopenmp filtro_gaussiano_mpi_rgb.c stb_impl.o -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi -o mainmpi
```
#### Com mpicc
```bash
mpicc -fopenmp filtro_gaussiano_mpi_rgb.c stb_impl.o -o mainmpi
```

**Versão sequencial:**
```bash
gcc -fopenmp filtro_gaussiano_serial_rgb.c stb_impl.o -o mainserial
```

### 🔹 Testes de corretude

```bash
gcc -fopenmp ./corretude/corretude.c stb_impl.o -o corretude
```

---

## ▶️ Execução

```bash
mpiexec -n (n de processos) mainmpi.exe
```
```bash
./mainserial
```
```bash
./corretude
```


---

## 🧠 Observações

- O projeto utiliza **MPI** para paralelização.
- O filtro Gaussiano é aplicado separadamente nos canais **RGB**.
- Foram implementadas versões **sequencial e multi-processos** para análise de desempenho.
- O teste de corretude garante que os resultados das execuções usando MPI e serial produzam os resultados adequados.

## Nota de Transparência sobre o Uso de IA: 
Declaro que este projeto contou com o auxílio das ferramentas de IA exclusivamente para as tarefas de 

• Revisão gramatical e tradução de trechos do relatório
• Auxílio na depuração (debugging) de blocos específicos de código
• Geração de scripts base para automação de testes de desempenho
• Estruturação inicial de ideias e revisão bibliográfica
• Estruturação de texto em formato LaTeX

Como autor(a), atesto que revisei, testei e validei criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos relatórios de desempenho e integridade acadêmica do material entregue.  

[Emanuel Humberto Menezes Cerqueira] – [01 de junho de 2026]  

[David Júnio Mariano dos Santos] – [01 de junho de 2026]