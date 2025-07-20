// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "publications",
          description: "(*) denotes equal contribution",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-notes",
          title: "notes",
          description: "A growing collection of my cool notes.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/notes/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "news-i-completed-my-master-s-degree-at-msu-with-highest-honors",
          title: 'I completed my Master’s degree at MSU with highest honors.',
          description: "",
          section: "News",},{id: "news-hello-from-bremen-my-phd-journey-starts-now",
          title: 'Hello from Bremen! My PhD Journey Starts Now.',
          description: "",
          section: "News",},{id: "notes-highly-accurate-protein-structure-prediction-with-alphafold",
          title: 'Highly accurate protein structure prediction with AlphaFold',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20generation/jumper2021highly/";
            },},{id: "notes-generating-novel-designable-and-diverse-protein-structures-by-equivariantly-diffusing-oriented-residue-clouds",
          title: 'Generating novel, designable, and diverse protein structures by equivariantly diffusing oriented residue clouds...',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20structure%20generation/lin2023generating/";
            },},{id: "notes-se-3-diffusion-model-with-application-to-protein-backbone-generation",
          title: 'SE (3) diffusion model with application to protein backbone generation',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20structure%20generation/yim2023se/";
            },},{id: "notes-de-novo-design-of-protein-structure-and-function-with-rfdiffusion",
          title: 'De novo design of protein structure and function with RFdiffusion',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20structure%20generation/watson2023novo/";
            },},{id: "notes-fast-protein-backbone-generation-with-se-3-flow-matching",
          title: 'Fast protein backbone generation with se (3) flow matching',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20structure%20generation/yim2023fast/";
            },},{id: "notes-one-step-diffusion-with-distribution-matching-distillation",
          title: 'One-step Diffusion with Distribution Matching Distillation',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/computer%20vision/diffusion%20distillation/yin2024one/";
            },},{id: "notes-generative-flows-on-discrete-state-spaces-enabling-multimodal-flows-with-applications-to-protein-co-design",
          title: 'Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design...',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20generation/campbell2024generative/";
            },},{id: "notes-accurate-structure-prediction-of-biomolecular-interactions-with-alphafold-3",
          title: 'Accurate structure prediction of biomolecular interactions with AlphaFold 3',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20generation/abramson2024accurate/";
            },},{id: "notes-improved-distribution-matching-distillation-for-fast-image-synthesis",
          title: 'Improved distribution matching distillation for fast image synthesis',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/computer%20vision/diffusion%20distillation/yin2024improved/";
            },},{id: "notes-regularized-distribution-matching-distillation-for-one-step-unpaired-image-to-image-translation",
          title: 'Regularized Distribution Matching Distillation for One-step Unpaired Image-to-Image Translation',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/computer%20vision/diffusion%20distillation/rakitin2024regularized/";
            },},{id: "notes-simulating-500-million-years-of-evolution-with-a-language-model",
          title: 'Simulating 500 million years of evolution with a language model',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20generation/hayes2025simulating/";
            },},{id: "notes-riemannian-score-based-generative-modelling",
          title: 'Riemannian Score-Based Generative Modelling',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/generative%20models/diffusion%20models/bortoli2022riemannian/";
            },},{id: "notes-multistate-and-functional-protein-design-using-rosettafold-sequence-space-diffusion",
          title: 'Multistate and functional protein design using RoseTTAFold sequence space diffusion',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/structural%20ensembles/lisanza2024multistate/";
            },},{id: "notes-scalable-emulation-of-protein-equilibrium-ensembles-with-generative-deep-learning",
          title: 'Scalable emulation of protein equilibrium ensembles with generative deep learning',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/structural%20ensembles/lewis2024scalable/";
            },},{id: "notes-inference-time-scaling-for-diffusion-models-beyond-scaling-denoising-steps",
          title: 'Inference-time scaling for diffusion models beyond scaling denoising steps',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/generative%20models/diffusion%20models/ma2025inference/";
            },},{id: "notes-continuous-diffusion-model-for-language-modeling",
          title: 'Continuous Diffusion Model for Language Modeling',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/natural%20language%20processing/text%20diffusion%20generation/jo2025continuous/";
            },},{id: "notes-aligning-protein-conformation-ensemble-generation-with-physical-feedback",
          title: 'Aligning Protein Conformation Ensemble Generation with Physical Feedback',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/structural%20ensembles/lu2025aligning/";
            },},{id: "notes-an-all-atom-generative-model-for-designing-protein-complexes",
          title: 'An All-Atom Generative Model for Designing Protein Complexes',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/bio-informatics/protein%20generation/chen2025all/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6D%65%73%68%63%68%61%6E%69%6E%6F%76.%76%69%61%63%68%65%73%6C%61%76@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/MeshchaninovViacheslav", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=FtnaZfsAAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
