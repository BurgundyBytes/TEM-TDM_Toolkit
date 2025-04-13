# src/utilities/asdm.py
````python
asdm_encode()
````

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none

    %% -------------------------------- Diagram Structure --------------------------------
    subgraph Input
        direction TB %% Arrange inputs vertically

        %% Input Nodes - Using refined labels (removed explicit '(optional)')
        In1[u: array-like<br/><span style='font-size:0.9em; color:#444'>Input Signal</span>]:::inputStyle
        In2[dt: float<br/><span style='font-size:0.9em; color:#444'>Signal Sample Time</span>]:::inputStyle
        In3[d_norm: float<br/><span style='font-size:0.9em; color:#444'>Encoder Threshold</span>]:::inputStyle
        In4[dte: float<br/><span style='font-size:0.9em; color:#444'>Encoding Sample Time</span>]:::optionalInputStyle
        In5[y: float<br/><span style='font-size:0.9em; color:#444'>Initial Integrator Value</span>]:::optionalInputStyle
        In6[interval: float<br/><span style='font-size:0.9em; color:#444'>Initial Time Since Spike</span>]:::optionalInputStyle
        In7[sgn: int<br/><span style='font-size:0.9em; color:#444'>Initial Integrator Sign</span>]:::optionalInputStyle
        In8[quad_method: str<br/><span style='font-size:0.9em; color:#444'>Integration Method</span>]:::optionalInputStyle

        %% Invisible Node to collect input arrows
        InputCollector( ):::collectorStyle
    end

    subgraph Process
        P[asdm_encode<br/><span style='font-size:0.9em; color:#444'>Performs ASDM Encoding</span>]:::processStyle
    end

    subgraph Output
        O[s: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Array of Spike Time Intervals</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    %% Connect all inputs to the invisible collector
    In1 --> InputCollector
    In2 --> InputCollector
    In3 --> InputCollector
    In4 --> InputCollector
    In5 --> InputCollector
    In6 --> InputCollector
    In7 --> InputCollector
    In8 --> InputCollector

    %% Connect the collector (representing all inputs) to the process
    InputCollector -- Parameters --> P

    %% Connect the process to the output
    P -- Encoded Spikes --> O
````

````python
asdm_decode()
````

````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none

    %% -------------------------------- Diagram Structure for asdm_decode --------------------------------
    subgraph Input
        direction TB %% Arrange inputs vertically

        %% Input Nodes based on asdm_decode arguments
        In1[s: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Spike Intervals</span>]:::inputStyle
        In2[dur: float<br/><span style='font-size:0.9em; color:#444'>Signal Duration</span>]:::inputStyle
        In3[dt: float<br/><span style='font-size:0.9em; color:#444'>Sampling Resolution </span>]:::inputStyle
        In4[bw: float<br/><span style='font-size:0.9em; color:#444'>Signal Bandwidth</span>]:::inputStyle
        In5[sgn: int<br/><span style='font-size:0.9em; color:#444'>Sign of First Spike</span>]:::optionalInputStyle 
        %% Optional (default: -1)

        %% Invisible Node to collect input arrows
        InputCollector( ):::collectorStyle
    end

    subgraph Process
        %% Process Node representing the function
        P[asdm_decode<br/><span style='font-size:0.9em; color:#444'>Decode ASDM Signal</span>]:::processStyle
    end

    subgraph Output
        %% Output Node based on the return value
        O[u_rec: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Recovered Signal</span>]:::outputStyle
    end

    %% -------------------------------- Connections --------------------------------
    %% Connect all inputs to the invisible collector
    In1 --> InputCollector
    In2 --> InputCollector
    In3 --> InputCollector
    In4 --> InputCollector
    In5 --> InputCollector

    %% Connect the collector (representing all inputs) to the process
    InputCollector -- Parameters --> P

    %% Connect the process to the output
    P -- Decoded Signal --> O
````

The functions work together ìn the following manner:
````mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#333', 'lineColor': '#666', 'fontSize': '14px' } } }%%
graph LR
    %% -------------------------------- Styles (Reusable Template Section) --------------------------------
    classDef inputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,color:#333,rx:4,ry:4 %% rx/ry for slightly rounded corners
    classDef processStyle fill:#fff,stroke:#888,stroke-width:1.5px,color:#000,font-weight:bold,rx:4,ry:4
    classDef outputStyle fill:#e8f5e9,stroke:#a5d6a7,stroke-width:1px,color:#333,rx:4,ry:4
    %% Optional input style: Same fill as input, but dashed border for clear visual cue
    classDef optionalInputStyle fill:#e3f2fd,stroke:#90caf9,stroke-width:1px,stroke-dasharray: 4 4,color:#333,rx:4,ry:4
    %% Invisible style for grouping nodes without visual clutter
    classDef collectorStyle fill:none,stroke:none
    %% Style for intermediate data linking processes
    classDef intermediateStyle fill:#fffde7,stroke:#fbc02d,stroke-width:1px,color:#333,rx:4,ry:4

    %% -------------------------------- Diagram Structure: Encode -> Decode Workflow --------------------------------

    %% == Encoding Stage ==
    subgraph Encoding Process
        direction TB

        %% --- Encoding Inputs ---
        Enc_In1[u: array-like<br/><span style='font-size:0.9em; color:#444'>Input Signal</span>]:::inputStyle
        Enc_In2[dt: float<br/><span style='font-size:0.9em; color:#444'>Signal Sample Time</span>]:::inputStyle
        Enc_In3[d_norm: float<br/><span style='font-size:0.9em; color:#444'>Encoder Threshold</span>]:::inputStyle
        Enc_In4[dte: float<br/><span style='font-size:0.9em; color:#444'>Encoding Sample Time</span>]:::optionalInputStyle
        Enc_In5[y: float<br/><span style='font-size:0.9em; color:#444'>Initial Integrator Value</span>]:::optionalInputStyle
        Enc_In6[interval: float<br/><span style='font-size:0.9em; color:#444'>Initial Time Since Spike</span>]:::optionalInputStyle
        Enc_In7[sgn: int<br/><span style='font-size:0.9em; color:#444'>Initial Integrator Sign</span>]:::optionalInputStyle
        Enc_In8[quad_method: str<br/><span style='font-size:0.9em; color:#444'>Integration Method</span>]:::optionalInputStyle

        %% --- Encoding Input Collector ---
        Enc_InputCollector( ):::collectorStyle

        %% --- Encoding Process Node ---
        P_Encode[asdm_encode<br/><span style='font-size:0.9em; color:#444'>Performs ASDM Encoding</span>]:::processStyle

        %% --- Connections within Encoding ---
        Enc_In1 --> Enc_InputCollector
        Enc_In2 --> Enc_InputCollector
        Enc_In3 --> Enc_InputCollector
        Enc_In4 --> Enc_InputCollector
        Enc_In5 --> Enc_InputCollector
        Enc_In6 --> Enc_InputCollector
        Enc_In7 --> Enc_InputCollector
        Enc_In8 --> Enc_InputCollector
        Enc_InputCollector -- Encode Params --> P_Encode
    end

    %% == Intermediate Data ==
    S_Link[s: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Spike Time Intervals<br/>]:::intermediateStyle

    %% == Decoding Stage ==
    subgraph Decoding Process
        direction TB

        %% --- Decoding Inputs (Specific to Decode) ---
        Dec_In1[dur: float<br/><span style='font-size:0.9em; color:#444'>Signal Duration</span>]:::inputStyle
        Dec_In2[dt: float<br/><span style='font-size:0.9em; color:#444'>Sampling Resolution</span>]:::inputStyle
        Dec_In3[bw: float<br/><span style='font-size:0.9em; color:#444'>Signal Bandwidth</span>]:::inputStyle
        Dec_In4[sgn: int<br/><span style='font-size:0.9em; color:#444'>Sign of First Spike</span>]:::optionalInputStyle

        %% --- Decoding Input Collector ---
        Dec_InputCollector( ):::collectorStyle

        %% --- Decoding Process Node ---
        P_Decode[asdm_decode<br/><span style='font-size:0.9em; color:#444'>Decode ASDM Signal</span>]:::processStyle

        %% --- Connections within Decoding ---
        %% Note: S_Link (from outside) connects to the collector
        Dec_In1 --> Dec_InputCollector
        Dec_In2 --> Dec_InputCollector
        Dec_In3 --> Dec_InputCollector
        Dec_In4 --> Dec_InputCollector
        Dec_InputCollector -- Decode Params & Spikes --> P_Decode
    end

    %% == Final Output ==
    O_Final[u_rec: np.ndarray<br/><span style='font-size:0.9em; color:#444'>Recovered Signal</span>]:::outputStyle


    %% -------------------------------- Overall Workflow Connections --------------------------------
    P_Encode -- Encoded Spikes --> S_Link
    S_Link --> Dec_InputCollector 
    %% Connect intermediate data to decode input collector
    P_Decode -- Decoded Signal --> O_Final
````