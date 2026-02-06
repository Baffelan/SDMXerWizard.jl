"""
Extension for GoogleGenAI support in SDMXerWizard

This extension is loaded automatically when GoogleGenAI is imported,
enabling Google Gemini models in SDMXerWizard workflows.
"""

module SDMXerWizardGoogleGenAIExt

using SDMXerWizard
using GoogleGenAI

# This extension ensures GoogleGenAI is available for PromptingTools.jl
# when using :google provider in SDMXerWizard functions

# No additional code needed - the mere loading of GoogleGenAI
# activates the PromptingTools.jl extension for Google models

end # module SDMXerWizardGoogleGenAIExt
