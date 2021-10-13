//===--- CodeGenAction.cpp - LLVM Code Generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenAction.h"
#include "CodeGenModule.h"
#include "CoverageMappingGen.h"
#include "CGCXXABI.h"
#include "MacroPPCallbacks.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/MemoryBufferCache.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <algorithm>
#include <cassert>
#include <cstdlib> // ::getenv
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <unordered_map>
using namespace clang;
using namespace llvm;

#define DEBUG_TYPE "clang-jit"

namespace {
// FIXME: This is copied from lib/Frontend/ASTUnit.cpp

/// Gathers information from ASTReader that will be used to initialize
/// a Preprocessor.
class ASTInfoCollector : public ASTReaderListener {
  Preprocessor &PP;
  ASTContext *Context;
  HeaderSearchOptions &HSOpts;
  PreprocessorOptions &PPOpts;
  LangOptions &LangOpt;
  std::shared_ptr<clang::TargetOptions> &TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> &Target;
  unsigned &Counter;
  bool InitializedLanguage = false;

public:
  ASTInfoCollector(Preprocessor &PP, ASTContext *Context,
                   HeaderSearchOptions &HSOpts, PreprocessorOptions &PPOpts,
                   LangOptions &LangOpt,
                   std::shared_ptr<clang::TargetOptions> &TargetOpts,
                   IntrusiveRefCntPtr<TargetInfo> &Target, unsigned &Counter)
      : PP(PP), Context(Context), HSOpts(HSOpts), PPOpts(PPOpts),
        LangOpt(LangOpt), TargetOpts(TargetOpts), Target(Target),
        Counter(Counter) {}

  bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
                           bool AllowCompatibleDifferences) override {
    if (InitializedLanguage)
      return false;

    LangOpt = LangOpts;
    InitializedLanguage = true;

    updated();
    return false;
  }

  bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                               StringRef SpecificModuleCachePath,
                               bool Complain) override {
    this->HSOpts = HSOpts;
    return false;
  }

  bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts, bool Complain,
                               std::string &SuggestedPredefines) override {
    this->PPOpts = PPOpts;
    return false;
  }

  bool ReadTargetOptions(const clang::TargetOptions &TargetOpts, bool Complain,
                         bool AllowCompatibleDifferences) override {
    // If we've already initialized the target, don't do it again.
    if (Target)
      return false;

    this->TargetOpts = std::make_shared<clang::TargetOptions>(TargetOpts);
    Target =
        TargetInfo::CreateTargetInfo(PP.getDiagnostics(), this->TargetOpts);

    updated();
    return false;
  }

  void ReadCounter(const serialization::ModuleFile &M,
                   unsigned Value) override {
    Counter = Value;
  }

private:
  void updated() {
    if (!Target || !InitializedLanguage)
      return;

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Target->adjust(LangOpt);

    // Initialize the preprocessor.
    PP.Initialize(*Target);

    if (!Context)
      return;

    // Initialize the ASTContext
    Context->InitBuiltinTypes(*Target);

    // Adjust printing policy based on language options.
    Context->setPrintingPolicy(PrintingPolicy(LangOpt));

    // We didn't have access to the comment options when the ASTContext was
    // constructed, so register them now.
    Context->getCommentCommandTraits().registerCommentOptions(
        LangOpt.CommentOpts);
  }
};

std::string getBadPipelinesFile() {
  char *ID = ::getenv("LLVM_JIT_BAD_PIPELINES_FILE");

  return ID;
}

void saveBadPipeline(const std::string &PassPipeline) {
  const std::string FileName = getBadPipelinesFile();
  llvm::errs() << "Writing to " << FileName << '\n';
  std::ofstream Output;

  Output.open(FileName, std::ios_base::app);
  if (!Output)
    report_fatal_error("JIT: couldn't open bad pipelines file\n");

  Output << PassPipeline << '\n';
  Output.close();
}

int CurrentKernel = 0;
std::string RandomPipeline = "";

void fatal(const std::string &PassPipeline="") {
  if (PassPipeline == "")
    report_fatal_error("Clang JIT failed!");
  else {
    llvm::errs() << "CurrentKernel: " << CurrentKernel;
    saveBadPipeline(PassPipeline);
    report_fatal_error("Clang JIT failed, saved bad pipeline!");
  }
}

// This is a variant of ORC's LegacyLookupFnResolver with a cutomized
// getResponsibilitySet behavior allowing us to claim responsibility for weak
// symbols in the loaded modules that we don't otherwise have.
// Note: We generally convert all IR level symbols to have strong linkage, but
// that won't cover everything (and especially doesn't cover the DW.ref.
// symbols created by the low-level EH logic on some platforms).
template <typename LegacyLookupFn>
class ClangLookupFnResolver final : public llvm::orc::SymbolResolver {
public:
  using ErrorReporter = std::function<void(Error)>;

  ClangLookupFnResolver(llvm::orc::ExecutionSession &ES,
                              LegacyLookupFn LegacyLookup,
                              ErrorReporter ReportError)
      : ES(ES), LegacyLookup(std::move(LegacyLookup)),
        ReportError(std::move(ReportError)) {}

  llvm::orc::SymbolNameSet
  getResponsibilitySet(const llvm::orc::SymbolNameSet &Symbols) final {
    llvm::orc::SymbolNameSet Result;

    for (auto &S : Symbols) {
      if (JITSymbol Sym = LegacyLookup(*S)) {
        // If the symbol exists elsewhere, and we have only a weak version,
        // then we're not responsible.
        continue;
      } else if (auto Err = Sym.takeError()) {
        ReportError(std::move(Err));
        return llvm::orc::SymbolNameSet();
      } else {
        Result.insert(S);
      }
    }

    return Result;
  }

  llvm::orc::SymbolNameSet
  lookup(std::shared_ptr<llvm::orc::AsynchronousSymbolQuery> Query,
                         llvm::orc::SymbolNameSet Symbols) final {
    return llvm::orc::lookupWithLegacyFn(ES, *Query, Symbols, LegacyLookup);
  }

private:
  llvm::orc::ExecutionSession &ES;
  LegacyLookupFn LegacyLookup;
  ErrorReporter ReportError;
};

template <typename LegacyLookupFn>
std::shared_ptr<ClangLookupFnResolver<LegacyLookupFn>>
createClangLookupResolver(llvm::orc::ExecutionSession &ES,
                          LegacyLookupFn LegacyLookup,
                          std::function<void(Error)> ErrorReporter) {
  return std::make_shared<ClangLookupFnResolver<LegacyLookupFn>>(
      ES, std::move(LegacyLookup), std::move(ErrorReporter));
}

class ClangJIT {
public:
  using ObjLayerT = llvm::orc::LegacyRTDyldObjectLinkingLayer;
  using CompileLayerT = llvm::orc::LegacyIRCompileLayer<ObjLayerT, llvm::orc::SimpleCompiler>;

  ClangJIT(DenseMap<StringRef, const void *> &LocalSymAddrs)
      : LocalSymAddrs(LocalSymAddrs),
        Resolver(createClangLookupResolver(
            ES,
            [this](const std::string &Name) {
              return findMangledSymbol(Name);
            },
            [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
        TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        ObjectLayer(ES,
                    [this](llvm::orc::VModuleKey) {
                      return ObjLayerT::Resources{
                          std::make_shared<SectionMemoryManager>(), Resolver};
                    }),
        CompileLayer(ObjectLayer, llvm::orc::SimpleCompiler(*TM)),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); }) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  ~ClangJIT() {
    // Run any destructors registered with __cxa_atexit.
    CXXRuntimeOverrides.runDestructors();

    // Run any IR destructors.
    for (auto &DtorRunner : IRStaticDestructorRunners)
      cantFail(DtorRunner.runViaLayer(CompileLayer));
  }

  llvm::TargetMachine &getTargetMachine() { return *TM; }

  llvm::orc::VModuleKey addModule(std::unique_ptr<llvm::Module> M) {
    // Record the static constructors and destructors. We have to do this before
    // we hand over ownership of the module to the JIT.
    std::vector<std::string> CtorNames, DtorNames;
    for (auto Ctor : llvm::orc::getConstructors(*M))
      if (Ctor.Func && !Ctor.Func->hasAvailableExternallyLinkage())
        CtorNames.push_back(mangle(Ctor.Func->getName()));
    for (auto Dtor : llvm::orc::getDestructors(*M))
      if (Dtor.Func && !Dtor.Func->hasAvailableExternallyLinkage())
        DtorNames.push_back(mangle(Dtor.Func->getName()));

    auto K = ES.allocateVModule();
    cantFail(CompileLayer.addModule(K, std::move(M)));
    ModuleKeys.push_back(K);

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    llvm::orc::LegacyCtorDtorRunner<CompileLayerT>
      CtorRunner(std::move(CtorNames), K);
    if (auto Err = CtorRunner.runViaLayer(CompileLayer)) {
      llvm::errs() << Err << "\n";
      fatal();
    }

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), K);

    return K;
  }

  void removeModule(llvm::orc::VModuleKey K) {
    ModuleKeys.erase(find(ModuleKeys, K));
    cantFail(CompileLayer.removeModule(K));
  }

  llvm::JITSymbol findSymbol(const std::string Name) {
    return findMangledSymbol(mangle(Name));
  }

private:
  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      llvm::raw_string_ostream MangledNameStream(MangledName);
      llvm::Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  llvm::JITSymbol findMangledSymbol(const std::string &Name) {
    for (auto H : make_range(ModuleKeys.rbegin(), ModuleKeys.rend()))
      if (auto Sym = CompileLayer.findSymbolIn(H, Name,
                                               /*ExportedSymbolsOnly*/ false))
        return Sym;

    if (auto Sym = CXXRuntimeOverrides.searchOverrides(Name))
      return Sym;

    auto LSAI = LocalSymAddrs.find(Name);
    if (LSAI != LocalSymAddrs.end())
      return llvm::JITSymbol(llvm::pointerToJITTargetAddress(LSAI->second),
                             llvm::JITSymbolFlags::Exported);

    // If we can't find the symbol in the JIT, try looking in the host process.
    if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
      return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);

#ifdef _WIN32
    // For Windows retry without "_" at beginning, as RTDyldMemoryManager uses
    // GetProcAddress and standard libraries like msvcrt.dll use names
    // with and without "_" (for example "_itoa" but "sin").
    if (Name.length() > 2 && Name[0] == '_')
      if (auto SymAddr =
              RTDyldMemoryManager::getSymbolAddressInProcess(Name.substr(1)))
        return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
#endif

    return nullptr;
  }

  DenseMap<StringRef, const void *> &LocalSymAddrs; 
  llvm::orc::ExecutionSession ES;
  std::shared_ptr<llvm::orc::SymbolResolver> Resolver;
  std::unique_ptr<llvm::TargetMachine> TM;
  const llvm::DataLayout DL;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  std::vector<llvm::orc::VModuleKey> ModuleKeys;

  llvm::orc::LegacyLocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<llvm::orc::LegacyCtorDtorRunner<CompileLayerT>>
    IRStaticDestructorRunners;
};

// Copied from CodeGenAction.cpp
class BackendConsumer;
class ClangDiagnosticHandler final : public DiagnosticHandler {
public:
  ClangDiagnosticHandler(const CodeGenOptions &CGOpts, BackendConsumer *BCon)
      : CodeGenOpts(CGOpts), BackendCon(BCon) {}

  bool handleDiagnostics(const DiagnosticInfo &DI) override;

  bool isAnalysisRemarkEnabled(StringRef PassName) const override {
    return (CodeGenOpts.OptimizationRemarkAnalysisPattern &&
            CodeGenOpts.OptimizationRemarkAnalysisPattern->match(PassName));
  }
  bool isMissedOptRemarkEnabled(StringRef PassName) const override {
    return (CodeGenOpts.OptimizationRemarkMissedPattern &&
            CodeGenOpts.OptimizationRemarkMissedPattern->match(PassName));
  }
  bool isPassedOptRemarkEnabled(StringRef PassName) const override {
    return (CodeGenOpts.OptimizationRemarkPattern &&
            CodeGenOpts.OptimizationRemarkPattern->match(PassName));
  }

  bool isAnyRemarkEnabled() const override {
    return (CodeGenOpts.OptimizationRemarkAnalysisPattern ||
            CodeGenOpts.OptimizationRemarkMissedPattern ||
            CodeGenOpts.OptimizationRemarkPattern);
  }

private:
  const CodeGenOptions &CodeGenOpts;
  BackendConsumer *BackendCon;
};

class BackendConsumer : public ASTConsumer {
  DiagnosticsEngine &Diags;
  BackendAction Action;
  const HeaderSearchOptions &HeaderSearchOpts;
  const CodeGenOptions &CodeGenOpts;
  const clang::TargetOptions &TargetOpts;
  const LangOptions &LangOpts;
  std::unique_ptr<raw_pwrite_stream> AsmOutStream;
  ASTContext *Context;
  std::string InFile;
  const PreprocessorOptions &PPOpts;
  LLVMContext &C;
  std::vector<std::unique_ptr<llvm::Module>> &DevLinkMods;
  CoverageSourceInfo *CoverageInfo;

  std::unique_ptr<CodeGenerator> Gen;

  void replaceGenerator() {
    Gen.reset(CreateLLVMCodeGen(Diags, InFile, HeaderSearchOpts, PPOpts,
                                CodeGenOpts, C, CoverageInfo));
  }

public:
  BackendConsumer(BackendAction Action, DiagnosticsEngine &Diags,
                  const HeaderSearchOptions &HeaderSearchOpts,
                  const PreprocessorOptions &PPOpts,
                  const CodeGenOptions &CodeGenOpts,
                  const clang::TargetOptions &TargetOpts,
                  const LangOptions &LangOpts, bool TimePasses,
                  const std::string &InFile,
                  std::unique_ptr<raw_pwrite_stream> OS, LLVMContext &C,
                  std::vector<std::unique_ptr<llvm::Module>> &DevLinkMods,
                  CoverageSourceInfo *CoverageInfo = nullptr)
      : Diags(Diags), Action(Action), HeaderSearchOpts(HeaderSearchOpts),
        CodeGenOpts(CodeGenOpts), TargetOpts(TargetOpts), LangOpts(LangOpts),
        AsmOutStream(std::move(OS)), Context(nullptr), InFile(InFile),
        PPOpts(PPOpts), C(C), DevLinkMods(DevLinkMods),
        CoverageInfo(CoverageInfo) { }

  llvm::Module *getModule() const { return Gen->GetModule(); }
  std::unique_ptr<llvm::Module> takeModule() {
    return std::unique_ptr<llvm::Module>(Gen->ReleaseModule());
  }

  CodeGenerator *getCodeGenerator() { return Gen.get(); }

  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override {
    Gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void Initialize(ASTContext &Ctx) override {
    replaceGenerator();
    Context = &Ctx;
    Gen->Initialize(Ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    Gen->HandleInlineFunctionDefinition(D);
  }

  void HandleInterestingDecl(DeclGroupRef D) override {
    HandleTopLevelDecl(D);
  }

  void HandleTranslationUnit(ASTContext &C) override {
      Gen->HandleTranslationUnit(C);

    // Silently ignore if we weren't initialized for some reason.
    if (!getModule())
      return;

    for (auto &BM : DevLinkMods) {
      std::unique_ptr<llvm::Module> M = llvm::CloneModule(*BM);
      M->setDataLayout(getModule()->getDataLayoutStr());
      M->setTargetTriple(getModule()->getTargetTriple());

      for (Function &F : *M)
        Gen->CGM().AddDefaultFnAttrs(F);

      bool Err = Linker::linkModules(
              *getModule(), std::move(M), llvm::Linker::Flags::LinkOnlyNeeded,
              [](llvm::Module &M, const llvm::StringSet<> &GVS) {
                internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
                  return !GV.hasName() || (GVS.count(GV.getName()) == 0);
                });
              });

      if (Err)
        fatal();
    }

  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    Gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    Gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    Gen->CompleteTentativeDefinition(D);
  }

  void AssignInheritanceModel(CXXRecordDecl *RD) override {
    Gen->AssignInheritanceModel(RD);
  }

  void HandleVTable(CXXRecordDecl *RD) override {
    Gen->HandleVTable(RD);
  }

  llvm::Error ReemitOptimized(llvm::Module *M, StringRef PassPipeline) {
    return EmitBackendOutput(Diags, HeaderSearchOpts, CodeGenOpts, TargetOpts,
                             LangOpts, Context->getTargetInfo().getDataLayout(),
                             M, Action, llvm::make_unique<llvm::buffer_ostream>(*AsmOutStream),
                             PassPipeline);
  }

  void EmitOptimized(StringRef PassPipeline) {
    // Copied from HandleTranslationUnit in CodeGenAction.cpp

    LLVMContext &Ctx = getModule()->getContext();
    LLVMContext::InlineAsmDiagHandlerTy OldHandler =
      Ctx.getInlineAsmDiagnosticHandler();
    void *OldContext = Ctx.getInlineAsmDiagnosticContext();
    Ctx.setInlineAsmDiagnosticHandler(InlineAsmDiagHandler, this);

    std::unique_ptr<DiagnosticHandler> OldDiagnosticHandler =
        Ctx.getDiagnosticHandler();
    Ctx.setDiagnosticHandler(llvm::make_unique<ClangDiagnosticHandler>(
      CodeGenOpts, this));
    Ctx.setDiagnosticsHotnessRequested(CodeGenOpts.DiagnosticsWithHotness);
    if (CodeGenOpts.DiagnosticsHotnessThreshold != 0)
      Ctx.setDiagnosticsHotnessThreshold(
          CodeGenOpts.DiagnosticsHotnessThreshold);

    std::unique_ptr<llvm::ToolOutputFile> OptRecordFile;
    if (!CodeGenOpts.OptRecordFile.empty()) {
      std::error_code EC;
      OptRecordFile = llvm::make_unique<llvm::ToolOutputFile>(
          CodeGenOpts.OptRecordFile, EC, sys::fs::F_None);
      if (EC) {
        Diags.Report(diag::err_cannot_open_file) <<
          CodeGenOpts.OptRecordFile << EC.message();
        return;
      }

      Ctx.setDiagnosticsOutputFile(
          llvm::make_unique<yaml::Output>(OptRecordFile->os()));

      if (CodeGenOpts.getProfileUse() != CodeGenOptions::ProfileNone)
        Ctx.setDiagnosticsHotnessRequested(true);
    }

    EmitBackendOutput(Diags, HeaderSearchOpts, CodeGenOpts, TargetOpts,
                      LangOpts, Context->getTargetInfo().getDataLayout(),
                      getModule(), Action,
                      llvm::make_unique<llvm::buffer_ostream>(*AsmOutStream),
                      PassPipeline);

    Ctx.setInlineAsmDiagnosticHandler(OldHandler, OldContext);

    Ctx.setDiagnosticHandler(std::move(OldDiagnosticHandler));

    if (OptRecordFile)
      OptRecordFile->keep();
  }

  // copied from CodeGenAction.cpp
  static void InlineAsmDiagHandler(const llvm::SMDiagnostic &SM,void *Context,
                                    unsigned LocCookie) {
    SourceLocation Loc = SourceLocation::getFromRawEncoding(LocCookie);
    ((BackendConsumer*)Context)->InlineAsmDiagHandler2(SM, Loc);
  }

  /// ConvertBackendLocation - Convert a location in a temporary llvm::SourceMgr
  /// buffer to be a valid FullSourceLoc.
  static FullSourceLoc ConvertBackendLocation(const llvm::SMDiagnostic &D,
                                              SourceManager &CSM) {
    // Get both the clang and llvm source managers.  The location is relative to
    // a memory buffer that the LLVM Source Manager is handling, we need to add
    // a copy to the Clang source manager.
    const llvm::SourceMgr &LSM = *D.getSourceMgr();

    // We need to copy the underlying LLVM memory buffer because llvm::SourceMgr
    // already owns its one and clang::SourceManager wants to own its one.
    const MemoryBuffer *LBuf =
    LSM.getMemoryBuffer(LSM.FindBufferContainingLoc(D.getLoc()));

    // Create the copy and transfer ownership to clang::SourceManager.
    // TODO: Avoid copying files into memory.
    std::unique_ptr<llvm::MemoryBuffer> CBuf =
        llvm::MemoryBuffer::getMemBufferCopy(LBuf->getBuffer(),
                                            LBuf->getBufferIdentifier());
    // FIXME: Keep a file ID map instead of creating new IDs for each location.
    FileID FID = CSM.createFileID(std::move(CBuf));

    // Translate the offset into the file.
    unsigned Offset = D.getLoc().getPointer() - LBuf->getBufferStart();
    SourceLocation NewLoc =
    CSM.getLocForStartOfFile(FID).getLocWithOffset(Offset);
    return FullSourceLoc(NewLoc, CSM);
  }

  /// InlineAsmDiagHandler2 - This function is invoked when the backend hits an
  /// error parsing inline asm.  The SMDiagnostic indicates the error relative to
  /// the temporary memory buffer that the inline asm parser has set up.
  void InlineAsmDiagHandler2(const llvm::SMDiagnostic &D,
                              SourceLocation LocCookie) {
    // There are a couple of different kinds of errors we could get here.  First,
    // we re-format the SMDiagnostic in terms of a clang diagnostic.

    // Strip "error: " off the start of the message string.
    StringRef Message = D.getMessage();
    if (Message.startswith("error: "))
      Message = Message.substr(7);

    // If the SMDiagnostic has an inline asm source location, translate it.
    FullSourceLoc Loc;
    if (D.getLoc() != SMLoc())
      Loc = ConvertBackendLocation(D, Context->getSourceManager());

    unsigned DiagID;
    switch (D.getKind()) {
    case llvm::SourceMgr::DK_Error:
      DiagID = diag::err_fe_inline_asm;
      break;
    case llvm::SourceMgr::DK_Warning:
      DiagID = diag::warn_fe_inline_asm;
      break;
    case llvm::SourceMgr::DK_Note:
      DiagID = diag::note_fe_inline_asm;
      break;
    case llvm::SourceMgr::DK_Remark:
      llvm_unreachable("remarks unexpected");
    }
    // If this problem has clang-level source location information, report the
    // issue in the source with a note showing the instantiated
    // code.
    if (LocCookie.isValid()) {
      Diags.Report(LocCookie, DiagID).AddString(Message);

      if (D.getLoc().isValid()) {
        DiagnosticBuilder B = Diags.Report(Loc, diag::note_fe_inline_asm_here);
        // Convert the SMDiagnostic ranges into SourceRange and attach them
        // to the diagnostic.
        for (const std::pair<unsigned, unsigned> &Range : D.getRanges()) {
          unsigned Column = D.getColumnNo();
          B << SourceRange(Loc.getLocWithOffset(Range.first - Column),
                          Loc.getLocWithOffset(Range.second - Column));
        }
      }
      return;
    }

    // Otherwise, report the backend issue as occurring in the generated .s file.
    // If Loc is invalid, we still need to report the issue, it just gets no
    // location info.
    Diags.Report(Loc, DiagID).AddString(Message);
  }

  const FullSourceLoc getBestLocationFromDebugLoc(
      const llvm::DiagnosticInfoWithLocationBase &D, bool &BadDebugInfo,
      StringRef &Filename, unsigned &Line, unsigned &Column) const {
    SourceManager &SourceMgr = Context->getSourceManager();
    FileManager &FileMgr = SourceMgr.getFileManager();
    SourceLocation DILoc;

    if (D.isLocationAvailable()) {
      D.getLocation(Filename, Line, Column);
      if (Line > 0) {
        const FileEntry *FE = FileMgr.getFile(Filename);
        if (!FE)
          FE = FileMgr.getFile(D.getAbsolutePath());
        if (FE) {
          // If -gcolumn-info was not used, Column will be 0. This upsets the
          // source manager, so pass 1 if Column is not set.
          DILoc = SourceMgr.translateFileLineCol(FE, Line, Column ? Column : 1);
        }
      }
      BadDebugInfo = DILoc.isInvalid();
    }

    // If a location isn't available, try to approximate it using the associated
    // function definition. We use the definition's right brace to differentiate
    // from diagnostics that genuinely relate to the function itself.
    FullSourceLoc Loc(DILoc, SourceMgr);
    if (Loc.isInvalid())
      if (const Decl *FD = Gen->GetDeclForMangledName(D.getFunction().getName()))
        Loc = FD->getASTContext().getFullLoc(FD->getLocation());

    if (DILoc.isInvalid() && D.isLocationAvailable())
      // If we were not able to translate the file:line:col information
      // back to a SourceLocation, at least emit a note stating that
      // we could not translate this location. This can happen in the
      // case of #line directives.
      Diags.Report(Loc, diag::note_fe_backend_invalid_loc)
          << Filename << Line << Column;

    return Loc;
  }


  void EmitOptimizationMessage(
    const llvm::DiagnosticInfoOptimizationBase &D, unsigned DiagID) {
    // We only support warnings and remarks.
    assert(D.getSeverity() == llvm::DS_Remark ||
          D.getSeverity() == llvm::DS_Warning);

    StringRef Filename;
    unsigned Line, Column;
    bool BadDebugInfo = false;
    FullSourceLoc Loc =
        getBestLocationFromDebugLoc(D, BadDebugInfo, Filename, Line, Column);

    std::string Msg;
    raw_string_ostream MsgStream(Msg);
    MsgStream << D.getMsg();

    if (D.getHotness())
      MsgStream << " (hotness: " << *D.getHotness() << ")";

    Diags.Report(Loc, DiagID)
        << AddFlagValue(D.getPassName())
        << MsgStream.str();

    if (BadDebugInfo)
      // If we were not able to translate the file:line:col information
      // back to a SourceLocation, at least emit a note stating that
      // we could not translate this location. This can happen in the
      // case of #line directives.
      Diags.Report(Loc, diag::note_fe_backend_invalid_loc)
          << Filename << Line << Column;
  }

  void OptimizationRemarkHandler(
      const llvm::DiagnosticInfoOptimizationBase &D) {
    // Without hotness information, don't show noisy remarks.
    if (D.isVerbose() && !D.getHotness())
      return;

    if (D.isPassed()) {
      // Optimization remarks are active only if the -Rpass flag has a regular
      // expression that matches the name of the pass name in \p D.
      if (CodeGenOpts.OptimizationRemarkPattern &&
          CodeGenOpts.OptimizationRemarkPattern->match(D.getPassName()))
        EmitOptimizationMessage(D, diag::remark_fe_backend_optimization_remark);
    } else if (D.isMissed()) {
      // Missed optimization remarks are active only if the -Rpass-missed
      // flag has a regular expression that matches the name of the pass
      // name in \p D.
      if (CodeGenOpts.OptimizationRemarkMissedPattern &&
          CodeGenOpts.OptimizationRemarkMissedPattern->match(D.getPassName()))
        EmitOptimizationMessage(
            D, diag::remark_fe_backend_optimization_remark_missed);
    } else {
      assert(D.isAnalysis() && "Unknown remark type");

      bool ShouldAlwaysPrint = false;
      if (auto *ORA = dyn_cast<llvm::OptimizationRemarkAnalysis>(&D))
        ShouldAlwaysPrint = ORA->shouldAlwaysPrint();

      if (ShouldAlwaysPrint ||
          (CodeGenOpts.OptimizationRemarkAnalysisPattern &&
          CodeGenOpts.OptimizationRemarkAnalysisPattern->match(D.getPassName())))
        EmitOptimizationMessage(
            D, diag::remark_fe_backend_optimization_remark_analysis);
    }
  }

  void OptimizationFailureHandler(
      const llvm::DiagnosticInfoOptimizationFailure &D) {
    EmitOptimizationMessage(D, diag::warn_fe_backend_optimization_failure);
  }

  #define ComputeDiagID(Severity, GroupName, DiagID)                             \
    do {                                                                         \
      switch (Severity) {                                                        \
      case llvm::DS_Error:                                                       \
        DiagID = diag::err_fe_##GroupName;                                       \
        break;                                                                   \
      case llvm::DS_Warning:                                                     \
        DiagID = diag::warn_fe_##GroupName;                                      \
        break;                                                                   \
      case llvm::DS_Remark:                                                      \
        llvm_unreachable("'remark' severity not expected");                      \
        break;                                                                   \
      case llvm::DS_Note:                                                        \
        DiagID = diag::note_fe_##GroupName;                                      \
        break;                                                                   \
      }                                                                          \
    } while (false)

  #define ComputeDiagRemarkID(Severity, GroupName, DiagID)                       \
    do {                                                                         \
      switch (Severity) {                                                        \
      case llvm::DS_Error:                                                       \
        DiagID = diag::err_fe_##GroupName;                                       \
        break;                                                                   \
      case llvm::DS_Warning:                                                     \
        DiagID = diag::warn_fe_##GroupName;                                      \
        break;                                                                   \
      case llvm::DS_Remark:                                                      \
        DiagID = diag::remark_fe_##GroupName;                                    \
        break;                                                                   \
      case llvm::DS_Note:                                                        \
        DiagID = diag::note_fe_##GroupName;                                      \
        break;                                                                   \
      }                                                                          \
    } while (false)

  bool InlineAsmDiagHandler(const llvm::DiagnosticInfoInlineAsm &D) {
    unsigned DiagID;
    ComputeDiagID(D.getSeverity(), inline_asm, DiagID);
    std::string Message = D.getMsgStr().str();

    // If this problem has clang-level source location information, report the
    // issue as being a problem in the source with a note showing the instantiated
    // code.
    SourceLocation LocCookie =
        SourceLocation::getFromRawEncoding(D.getLocCookie());
    if (LocCookie.isValid())
      Diags.Report(LocCookie, DiagID).AddString(Message);
    else {
      // Otherwise, report the backend diagnostic as occurring in the generated
      // .s file.
      // If Loc is invalid, we still need to report the diagnostic, it just gets
      // no location info.
      FullSourceLoc Loc;
      Diags.Report(Loc, DiagID).AddString(Message);
    }
    // We handled all the possible severities.
    return true;
  }

  bool StackSizeDiagHandler(const llvm::DiagnosticInfoStackSize &D) {
    if (D.getSeverity() != llvm::DS_Warning)
      // For now, the only support we have for StackSize diagnostic is warning.
      // We do not know how to format other severities.
      return false;

    if (const Decl *ND = Gen->GetDeclForMangledName(D.getFunction().getName())) {
      // FIXME: Shouldn't need to truncate to uint32_t
      Diags.Report(ND->getASTContext().getFullLoc(ND->getLocation()),
                  diag::warn_fe_frame_larger_than)
        << static_cast<uint32_t>(D.getStackSize()) << Decl::castToDeclContext(ND);
      return true;
    }

    return false;
  }

  void UnsupportedDiagHandler(
      const llvm::DiagnosticInfoUnsupported &D) {
    // We only support errors.
    assert(D.getSeverity() == llvm::DS_Error);

    StringRef Filename;
    unsigned Line, Column;
    bool BadDebugInfo = false;
    FullSourceLoc Loc =
        getBestLocationFromDebugLoc(D, BadDebugInfo, Filename, Line, Column);

    Diags.Report(Loc, diag::err_fe_backend_unsupported) << D.getMessage().str();

    if (BadDebugInfo)
      // If we were not able to translate the file:line:col information
      // back to a SourceLocation, at least emit a note stating that
      // we could not translate this location. This can happen in the
      // case of #line directives.
      Diags.Report(Loc, diag::note_fe_backend_invalid_loc)
          << Filename << Line << Column;
  }

  void DiagnosticHandlerImpl(const llvm::DiagnosticInfo &DI) {
    unsigned DiagID = diag::err_fe_inline_asm;
    llvm::DiagnosticSeverity Severity = DI.getSeverity();
    // Get the diagnostic ID based.
    switch (DI.getKind()) {
    case llvm::DK_InlineAsm:
      if (InlineAsmDiagHandler(cast<DiagnosticInfoInlineAsm>(DI)))
        return;
      ComputeDiagID(Severity, inline_asm, DiagID);
      break;
    case llvm::DK_StackSize:
      if (StackSizeDiagHandler(cast<DiagnosticInfoStackSize>(DI)))
        return;
      ComputeDiagID(Severity, backend_frame_larger_than, DiagID);
      break;
    case DK_Linker:
      assert(getModule());
      // FIXME: stop eating the warnings and notes.
      if (Severity != DS_Error)
        return;
      DiagID = diag::err_fe_cannot_link_module;
    break;
    case llvm::DK_OptimizationRemark:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<OptimizationRemark>(DI));
      return;
    case llvm::DK_OptimizationRemarkMissed:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<OptimizationRemarkMissed>(DI));
      return;
    case llvm::DK_OptimizationRemarkAnalysis:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<OptimizationRemarkAnalysis>(DI));
      return;
    case llvm::DK_OptimizationRemarkAnalysisFPCommute:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<OptimizationRemarkAnalysisFPCommute>(DI));
      return;
    case llvm::DK_OptimizationRemarkAnalysisAliasing:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<OptimizationRemarkAnalysisAliasing>(DI));
      return;
    case llvm::DK_MachineOptimizationRemark:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<MachineOptimizationRemark>(DI));
      return;
    case llvm::DK_MachineOptimizationRemarkMissed:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<MachineOptimizationRemarkMissed>(DI));
      return;
    case llvm::DK_MachineOptimizationRemarkAnalysis:
      // Optimization remarks are always handled completely by this
      // handler. There is no generic way of emitting them.
      OptimizationRemarkHandler(cast<MachineOptimizationRemarkAnalysis>(DI));
      return;
    case llvm::DK_OptimizationFailure:
      // Optimization failures are always handled completely by this
      // handler.
      OptimizationFailureHandler(cast<DiagnosticInfoOptimizationFailure>(DI));
      return;
    case llvm::DK_Unsupported:
      UnsupportedDiagHandler(cast<DiagnosticInfoUnsupported>(DI));
      return;
    default:
      // Plugin IDs are not bound to any value as they are set dynamically.
      ComputeDiagRemarkID(Severity, backend_plugin, DiagID);
      break;
    }
    std::string MsgStorage;
    {
      raw_string_ostream Stream(MsgStorage);
      DiagnosticPrinterRawOStream DP(Stream);
      DI.print(DP);
    }

    if (DiagID == diag::err_fe_cannot_link_module) {
      Diags.Report(diag::err_fe_cannot_link_module)
          << getModule()->getModuleIdentifier() << MsgStorage;
      return;
    }

    // Report the backend message using the usual diagnostic mechanism.
    FullSourceLoc Loc;
    Diags.Report(Loc, DiagID).AddString(MsgStorage);

  }

};

// copied from CodeGenAction.cpp
bool ClangDiagnosticHandler::handleDiagnostics(const DiagnosticInfo &DI) {
  BackendCon->DiagnosticHandlerImpl(DI);
  return true;
}

class JFIMapDeclVisitor : public RecursiveASTVisitor<JFIMapDeclVisitor> {
  DenseMap<unsigned, FunctionDecl *> &Map;

public:
  explicit JFIMapDeclVisitor(DenseMap<unsigned, FunctionDecl *> &M)
    : Map(M) { }

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitFunctionDecl(const FunctionDecl *D) {
    if (auto *A = D->getAttr<JITFuncInstantiationAttr>())
      Map[A->getId()] = const_cast<FunctionDecl *>(D);
    return true;
  }
};

class JFICSMapDeclVisitor : public RecursiveASTVisitor<JFICSMapDeclVisitor> {
  DenseMap<unsigned, FunctionDecl *> &Map;
  SmallVector<FunctionDecl *, 1> CurrentFD;

public:
  explicit JFICSMapDeclVisitor(DenseMap<unsigned, FunctionDecl *> &M)
    : Map(M) { }

  bool TraverseFunctionDecl(FunctionDecl *FD) {
    CurrentFD.push_back(FD);
    bool Continue =
      RecursiveASTVisitor<JFICSMapDeclVisitor>::TraverseFunctionDecl(FD);
    CurrentFD.pop_back();

    return Continue;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    auto *FD = dyn_cast<FunctionDecl>(E->getDecl());
    if (!FD)
      return true;

    auto *A = FD->getAttr<JITFuncInstantiationAttr>();
    if (!A)
      return true;

    Map[A->getId()] = CurrentFD.back();

    return true;
  }
};

unsigned LastUnique = 0;
std::unique_ptr<llvm::LLVMContext> LCtx;

enum JITPassType {
  ModulePass = 0,
  CGSCCPass,
  FunctionPass,
  LoopPass
};

struct JITPass {
  StringRef Name;
  JITPassType PassType;

  JITPass(StringRef Name, JITPassType Type) : Name(Name), PassType(Type) {}
};

using JITMultiPass = std::vector<std::unique_ptr<JITPass>>;

class JITPipeline {
public:
  std::vector<std::unique_ptr<JITMultiPass>> Passes;
  JITPipeline() {};

  void addMultiPass(const JITMultiPass& MP) {
    auto NewMP = llvm::make_unique<JITMultiPass>();
    for (const auto& JP : MP) {
      NewMP->push_back(llvm::make_unique<JITPass>(JP->Name, JP->PassType));
    }

    Passes.push_back(std::move(NewMP));
  }

  void addPass(const JITPass &JP) {
    auto NewMP = llvm::make_unique<JITMultiPass>();
    NewMP->push_back(llvm::make_unique<JITPass>(JP.Name, JP.PassType));
    Passes.push_back(std::move(NewMP));
  }

  void removePass(int Index) {
    Passes.erase(Passes.begin() + Index);
  }

  void shuffle(int VariantIdx) {
    auto RNG = std::default_random_engine {};
    RNG.seed(VariantIdx);
    std::shuffle(std::begin(Passes), std::end(Passes), RNG);
  }

  void buildPassPipeline(PassBuilder::OptimizationLevel Level) {
    // Passes and order copied from
    // PassBuilder::buildModuleSimplificationPipeline

    addPass({"forceattrs", ModulePass});
    addPass({"inferattrs", ModulePass});
    addPass({"simplify-cfg", FunctionPass});
    addPass({"sroa", FunctionPass});
    addPass({"early-cse", FunctionPass});
    addPass({"lower-expect", FunctionPass});
    if (Level == PassBuilder::O3)
      addPass({"callsite-splitting", FunctionPass});

    addPass({"ipsccp", ModulePass});
    addPass({"called-value-propagation", ModulePass});
    addPass({"globalopt", ModulePass});
    addPass({"mem2reg", FunctionPass});
    addPass({"deadargelim", ModulePass});
    addPass({"instcombine", FunctionPass});
    addPass({"simplify-cfg", FunctionPass});

    addPass({"require<globals-aa>", ModulePass});
    addPass({"require<profile-summary>", ModulePass});

    addPass({"inline", CGSCCPass}); // TODO: need to pass args to inline
    addPass({"function-attrs", CGSCCPass});

    if (Level == PassBuilder::O3)
      addPass({"argpromotion", CGSCCPass});

    // Passes and order copied from
    // PassBuilder::buildFunctionSimplificationPipeline

    addPass({"sroa", FunctionPass});
    addPass({"early-cse", FunctionPass}); // TODO: this takes a boolean argument

    // TODO: GVNHoist? GVNSink?

    addPass({"speculative-execution", FunctionPass});
    addPass({"jump-threading", FunctionPass});
    addPass({"correlated-propagation", FunctionPass});
    addPass({"simplify-cfg", FunctionPass});

    if (Level == PassBuilder::O3)
      addPass({"aggressive-instcombine", FunctionPass});

    addPass({"instcombine", FunctionPass});
    addPass({"libcalls-shrinkwrap", FunctionPass});

    addPass({"tailcallelim", FunctionPass});
    addPass({"simplify-cfg", FunctionPass});

    addPass({"reassociate", FunctionPass});

    JITMultiPass MP1;
    MP1.push_back(llvm::make_unique<JITPass>("require<opt-remark-emit>", FunctionPass));
    MP1.push_back(llvm::make_unique<JITPass>("loop-instsimplify", LoopPass));
    addMultiPass(MP1);

    addPass({"simplify-cfg", LoopPass});
    addPass({"rotate", LoopPass});
    addPass({"unswitch", LoopPass});

    addPass({"simplify-cfg", FunctionPass});
    addPass({"instcombine", FunctionPass});

    addPass({"indvars", LoopPass});
    addPass({"loop-idiom", LoopPass});
    addPass({"loop-deletion", LoopPass});

    if (Level != PassBuilder::O1) {
      addPass({"mldst-motion", FunctionPass});
      addPass({"gvn", FunctionPass});
    }

    addPass({"memcpyopt", FunctionPass});
    addPass({"sccp", FunctionPass});
    addPass({"bdce", FunctionPass});
    addPass({"instcombine", FunctionPass});

    addPass({"jump-threading", FunctionPass});
    addPass({"correlated-propagation", FunctionPass});
    addPass({"dse", FunctionPass});

    JITMultiPass MP2;
    MP2.push_back(llvm::make_unique<JITPass>("require<opt-remark-emit>", FunctionPass));
    MP2.push_back(llvm::make_unique<JITPass>("licm", LoopPass));
    addMultiPass(MP2);

    addPass({"adce", FunctionPass});
    addPass({"simplify-cfg", FunctionPass});
    addPass({"instcombine", FunctionPass});

    // TODO: devirt pass?

    // Passes and order copied from
    // PassBuilder::buildModuleOptimizationPipeline

    addPass({"globalopt", ModulePass});
    addPass({"globaldce", ModulePass});

    // TODO: partial inlining?

    addPass({"elim-avail-extern", ModulePass}); // Removing this causes an error
    addPass({"rpo-functionattrs", ModulePass});
    addPass({"require<globals-aa>", ModulePass});

    addPass({"float2int", FunctionPass});
    addPass({"rotate", LoopPass});
    addPass({"loop-distribute", FunctionPass});
    addPass({"loop-vectorize", FunctionPass});
    addPass({"loop-load-elim", FunctionPass});
    addPass({"instcombine", FunctionPass});
    addPass({"simplify-cfg", FunctionPass}); // TODO: need to pass args here
    addPass({"slp-vectorizer", FunctionPass});
    addPass({"instcombine", FunctionPass});

    JITMultiPass MP3;
    MP3.push_back(llvm::make_unique<JITPass>("require<opt-remark-emit>", FunctionPass));
    MP3.push_back(llvm::make_unique<JITPass>("licm", LoopPass));
    addMultiPass(MP3);

    // TODO: enableUnrollAndJam?

    addPass({"unroll", FunctionPass}); // TODO: need to pass args here
    addPass({"transform-warning", FunctionPass});
    addPass({"instcombine", FunctionPass});

    JITMultiPass MP4;
    MP4.push_back(llvm::make_unique<JITPass>("require<opt-remark-emit>", FunctionPass));
    MP4.push_back(llvm::make_unique<JITPass>("licm", LoopPass));
    addMultiPass(MP4);

    addPass({"alignment-from-assumptions", FunctionPass});
    addPass({"loop-sink", FunctionPass});
    addPass({"instsimplify", FunctionPass});
    addPass({"div-rem-pairs", FunctionPass});
    addPass({"simplify-cfg", FunctionPass});
    addPass({"spec-phis", FunctionPass});

    addPass({"cg-profile", ModulePass});

    addPass({"globaldce", ModulePass});
    // addPass({"constmerge", ModulePass});

    addPass({"canonicalize-aliases", ModulePass});
    addPass({"name-anon-globals", ModulePass});
  }

  std::string getPassTypeString(JITPassType Type) {
    if (Type == ModulePass)
      return "module";
    if (Type == CGSCCPass)
      return "cgscc";
    if (Type == FunctionPass)
      return "function";
    if (Type == LoopPass)
      return "loop";
  }

  // Convert vector of MultiPasses to Passes
  std::vector<std::unique_ptr<JITPass>> flattenPipeline() {
    size_t total_size = 0;
    for (auto const& MP: Passes){
      total_size += MP->size();
    }

    std::vector<std::unique_ptr<JITPass>> Pipeline;
    Pipeline.reserve(total_size);

    for (auto& MP: Passes) {
        std::move(MP->begin(), MP->end(), std::back_inserter(Pipeline));
    }

    return Pipeline;
  }

  std::string toString() {
    auto FlattenedPasses = flattenPipeline();

    JITPassType CurrentType = ModulePass;
    std::string Pipeline = "module(";

    bool PassesOpen[4] = {true, false, false, false};

    size_t index = 0;
    for (const auto& Pass : FlattenedPasses) {
      if (Pass->PassType != CurrentType) {
        if (Pass->PassType > CurrentType) {
          // No Module/CGSCC to Loop pass adaptor, so we add "function" manually
          if (Pass->PassType == LoopPass && (PassesOpen[ModulePass] || PassesOpen[CGSCCPass]) && !PassesOpen[FunctionPass]) {
            Pipeline += "function(";
            PassesOpen[FunctionPass] = true;
          }

          Pipeline += getPassTypeString(Pass->PassType) + "(";
          PassesOpen[Pass->PassType] = true;
        }

        else if (Pass->PassType < CurrentType) {
          if (Pipeline.back() == ',')
            Pipeline.pop_back();

          int Parens = 0;
          for (int i = Pass->PassType + 1; i <= CurrentType; i++) {
            if (PassesOpen[i]) {
              Parens++;
              PassesOpen[i] = false;
            }
          }

          while (Parens > 0) {
            Pipeline += ")";
            Parens--;
          }

          Pipeline += ',';

          if (!PassesOpen[Pass->PassType]) {
            Pipeline += getPassTypeString(Pass->PassType) + "(";
            PassesOpen[Pass->PassType] = true;
          }
        }
      }

      CurrentType = Pass->PassType;
      Pipeline += Pass->Name.str();

      if (index == FlattenedPasses.size() - 1) {
        int Parens = 0;
        for (const char c: Pipeline) {
          if (c == '(')
            Parens++;
          else if (c == ')')
            Parens--;
        }

        while (Parens > 0) {
          Pipeline += ")";
          Parens--;
        }
      }
      else
        Pipeline += ",";

      index++;
    }

    return Pipeline;
  }
};

std::set<std::string> BadPipelines;

void readBadPipelines() {
  const std::string FileName = getBadPipelinesFile();
  llvm::errs() << "Reading from " << FileName << '\n';
  std::ifstream Input;

  Input.open(FileName);
  if (!Input)
    return;

  if (Input.is_open()) {
    std::string line;
    while (std::getline(Input, line)) {
      BadPipelines.insert(line);
    }
  }

  Input.close();
}

class RandomPipelineBuilder {
public:
  RandomPipelineBuilder(
      std::shared_ptr<BackendConsumer> &Consumer, size_t PipelineSize, size_t NumPasses,
      std::vector<std::unique_ptr<JITMultiPass>> &&PassMap)
    : Consumer(Consumer), PipelineSize(PipelineSize), NumPasses(NumPasses),
      PassMap(std::move(PassMap)) { }

  std::string Run() {
    int iterations = 0;
    bool isIn = true;
    std::string StringPipeline = "";

    while (isIn && iterations < 10) {
      iterations++;
      StringPipeline = generateRandomPipeline();
      isIn = BadPipelines.find(StringPipeline) != BadPipelines.end();
    }

    if (iterations == 10)
      report_fatal_error("Max iterations exceeded during random search");

    return StringPipeline;
  }

private:
  std::shared_ptr<BackendConsumer> Consumer;
  size_t PipelineSize;
  size_t NumPasses;
  std::vector<std::unique_ptr<JITMultiPass>> PassMap; // Maps an integer to a Pass

  std::string generateRandomPipeline() {
    std::random_device RandomDevice;
    std::mt19937 Engine{RandomDevice()};
    std::uniform_int_distribution<int> Distribution{0, static_cast<int>(NumPasses) - 1};

    auto Gen = [&Distribution, &Engine](){ return Distribution(Engine); };

    std::vector<int> Pipeline(PipelineSize);
    std::generate(std::begin(Pipeline), std::end(Pipeline), Gen);

    JITPipeline JP;

    for (int Pass : Pipeline)
      JP.addMultiPass(*PassMap[Pass]);

    return JP.toString();
  }
};

class GeneticPipelineBuilder {
public:
  GeneticPipelineBuilder(
      std::shared_ptr<BackendConsumer> &Consumer, size_t PopulationSize,
      size_t ChromosomeSize, size_t GenePoolSize, std::vector<std::unique_ptr<JITMultiPass>> &&PassMap)
    : Consumer(Consumer), PopulationSize(PopulationSize), ChromosomeSize(ChromosomeSize),
      GenePoolSize(GenePoolSize), Chromosomes(PopulationSize, std::vector<int>(ChromosomeSize)),
      NewChromosomes(PopulationSize, std::vector<int>(ChromosomeSize)),
      Probabilities(PopulationSize, 0.0), CumulativeProbabilities(PopulationSize, 0.0),
      Fitness(PopulationSize, 0.0), PreviousFitness(PopulationSize, 0.0),
      Mod(llvm::CloneModule(*Consumer->getModule())), PassMap(std::move(PassMap)), O3Instructions(0),
      O3LoopsVectorized(0) {
        JITPipeline JP;
        JP.buildPassPipeline(PassBuilder::OptimizationLevel::O3);
        std::unique_ptr<llvm::Module> M = llvm::CloneModule(*Mod);
        Consumer->ReemitOptimized(M.get(), JP.toString());
        O3Instructions = M->getInstructionCount();

        auto Stats = llvm::GetStatistics();
        auto VectorizeStat = std::find_if(Stats.begin(), Stats.end(),
                                          [](const std::pair<StringRef, unsigned>& S) {
                                          return std::get<0>(S) == "LoopsVectorized"; } );

        if (VectorizeStat != Stats.end())
          O3LoopsVectorized = std::get<1>(*VectorizeStat);
        else
          O3LoopsVectorized = 0;
      }

  std::string Run() {
    // taken mostly from https://arxiv.org/pdf/1308.4675.pdf,
    // and partly from https://www.whitman.edu/Documents/Academics/Mathematics/2014/carrjk.pdf
    // and https://towardsdatascience.com/genetic-algorithm-explained-step-by-step-65358abe2bf

    buildInitialPopulation();
    evaluateFitness();

    llvm::errs() << "Start ********\n";
    printStats();
    llvm::errs() << "**************\n";

    int i = 0;
    for (i = 0; i < 100; i++) {
      evaluateFitness();
      computeProbabilities();
      selectNewChromosomes();
      mateChromosomes();
      mutateGenes();
      if (i == 0 || i == 49 || i == 99) {
        llvm::errs() << "Index " << i << " ********\n";
        printStats();
        // printChromosomes();
        llvm::errs() << "*****************\n";
        PreviousFitness = Fitness;
      }
    }

    llvm::errs() << "O3 num instructions: " << O3Instructions << '\n';
    auto min = std::max_element(std::begin(Fitness), std::end(Fitness));
    llvm::errs() << "Num instrs " << (unsigned) (1 / Fitness[min - Fitness.begin()]) << '\n';

    // llvm::errs() << "O3 Loops Vectorized: " << O3LoopsVectorized << '\n';
    // auto min = std::max_element(std::begin(Fitness), std::end(Fitness));
    // llvm::errs() << "Loops Vectorized " << (unsigned) (Fitness[min - Fitness.begin()]) << '\n';

    return buildPipeline(Chromosomes[min - Fitness.begin()]);
  }

private:
  std::shared_ptr<BackendConsumer> Consumer;
  size_t PopulationSize;
  size_t ChromosomeSize;
  size_t GenePoolSize;
  std::vector<std::vector<int>> Chromosomes;
  std::vector<std::vector<int>> NewChromosomes;
  std::vector<double> Probabilities;
  std::vector<double> CumulativeProbabilities;
  std::vector<double> Fitness;
  std::vector<double> PreviousFitness; // Needed for stats
  std::unique_ptr<llvm::Module> Mod;
  std::vector<std::unique_ptr<JITMultiPass>> PassMap; // Maps an integer to a Pass
  unsigned O3Instructions;
  unsigned O3LoopsVectorized;

  static constexpr double MutationRate = 0.001;

  void buildInitialPopulation() {
    std::random_device RandomDevice;
    std::mt19937 Engine{RandomDevice()};
    std::uniform_int_distribution<int> Distribution{0, static_cast<int>(GenePoolSize) - 1};

    auto Gen = [&Distribution, &Engine](){ return Distribution(Engine); };

    for (auto &C : Chromosomes)
      std::generate(std::begin(C), std::end(C), Gen);
  }

  void printChromosomes(bool PrintNew=false) {
    size_t i = 0;

    for (const auto &C: Chromosomes) {
      llvm::errs() << i << ": ";
      for (const auto Gene: C)
        llvm::errs() << Gene << ", ";

      llvm::errs() << "; P[" << i << "] = " << Probabilities[i] << '\n';
      i++;
    }

    if (PrintNew) {
      size_t i = 0;

      for (const auto &C: NewChromosomes) {
        llvm::errs() << i << ": ";
        for (const auto Gene: C)
          llvm::errs() << Gene;

        llvm::errs() << '\n';
        i++;
      }
    }
  }

  void printStats() {
    double TotalFitness = 0;
    for (size_t i = 0; i < PopulationSize; i++)
      TotalFitness += Fitness[i];

    llvm::errs() << "Average " << (unsigned) (PopulationSize / TotalFitness) << '\n';

    auto max = std::max_element(std::begin(Fitness), std::end(Fitness));
    auto min = std::min_element(std::begin(Fitness), std::end(Fitness));
    llvm::errs() << "Min instructions " << (unsigned) (1 / *max) << '\n';
    llvm::errs() << "Max instructions " << (unsigned) (1 / *min) << '\n';

    // llvm::errs() << "Average " << (unsigned) (TotalFitness / PopulationSize) << '\n';

    // auto max = std::max_element(std::begin(Fitness), std::end(Fitness));
    // auto min = std::min_element(std::begin(Fitness), std::end(Fitness));
    // llvm::errs() << "Max Vectorized " << (unsigned) (*max) << '\n';
    // llvm::errs() << "Min Vectorized " << (unsigned) (*min) << '\n';

    unsigned NumImproved = 0;
    unsigned NumWorsened = 0;
    unsigned NumSame = 0;
    unsigned NumFailed = 0;
    for (size_t i = 0; i < PopulationSize; i++) {
      if (Fitness[i] == 0)
        NumFailed++;
      else if (Fitness[i] > PreviousFitness[i])
        NumImproved++;
      else if (Fitness[i] < PreviousFitness[i])
        NumWorsened++;
      else if (Fitness[i] == PreviousFitness[i])
        NumSame++;
    }

    llvm::errs() << "Improved: " << NumImproved << "; Worsened: " << NumWorsened
                 << "; Same: " << NumSame << "; Failed: " << NumFailed << '\n';
  }

  void computeProbabilities() {
    double TotalFitness = 0;
    for (size_t i = 0; i < PopulationSize; i++)
      TotalFitness += Fitness[i];

    for (size_t i = 0; i < PopulationSize; i++) {
      Probabilities[i] = Fitness[i] / (double) TotalFitness;

      if (i == 0)
        CumulativeProbabilities[i] = Probabilities[i];
      else
        CumulativeProbabilities[i] = CumulativeProbabilities[i - 1] + Probabilities[i];
    }
  }

  std::string buildPipeline(const std::vector<int> &Chromosome) {
    JITPipeline JP;

    for (int Gene: Chromosome)
      JP.addMultiPass(*PassMap[Gene]);

    return JP.toString();
  }

  void evaluateFitness() {
    int index = 0;
    for (const auto &C: Chromosomes) {
      Fitness[index] = calculateChromosomeFitness(C);
      index++;
    }
  }

  double calculateChromosomeFitness(const std::vector<int>& C) {
    std::unique_ptr<llvm::Module> M = llvm::CloneModule(*Mod);

    double F;
    auto Pipeline = buildPipeline(C);
    // llvm::errs() << "Pipeline: \n";
    // for (const int n : C)
    //   llvm::errs() << n << " ";
    // llvm::errs() << "\n************\n";
    // llvm::errs() << Pipeline << '\n';

    if (auto Err = Consumer->ReemitOptimized(M.get(), buildPipeline(C)))
      F = 0;
    else {
      F = getStatValue("LoopsVectorized");
      F += getStatValue("NumBranchOpts");
      F += getStatValue("NumConstantsHoisted");
      F += getStatValue("NumInlined");
      F += getStatValue("NumCombined");
      F += getStatValue("NumHoisted");
      F += getStatValue("NumMovedLoads");
      F += getStatValue("NumUnrolled");
      F += getStatValue("LoopsVectorized");
      // F += getStatValue("NumSpills");
      F += getStatValue("NumVectorized");
    }

    llvm::ResetStatistics();
    return F;
  }

  double getStatValue(StringRef StatName) {
    auto Stats = llvm::GetStatistics();
    auto Stat = std::find_if(Stats.begin(), Stats.end(),
                              [](const std::pair<StringRef, unsigned>& S) {
                                  return std::get<0>(S) == "LoopsVectorized"; } );

    double Value;
    if (Stat != Stats.end())
      Value = std::get<1>(*Stat);
    else
      Value = 0;

    return Value;
  }

  void selectNewChromosomes() {
    std::vector<double> Random(PopulationSize);

    std::random_device RandomDevice;
    std::mt19937 Engine{RandomDevice()};
    std::uniform_real_distribution<double> Distribution{0, 1};

    auto Gen = [&Distribution, &Engine](){ return Distribution(Engine); };

    std::generate(std::begin(Random), std::end(Random), Gen);

    for (size_t i = 0; i < PopulationSize; i++) {
      for (size_t j = 0; j < PopulationSize; j++) {
        NewChromosomes[i] = Chromosomes[j];
        if (Random[i] < CumulativeProbabilities[j]) {
          break;
        }
      }
    }
  }

  void mateChromosomes() {
    std::random_device RandomDevice;
    std::mt19937 Engine{RandomDevice()};
    std::uniform_int_distribution<int> Distribution{1, static_cast<int>(ChromosomeSize) - 2}; // Select the index for the crossover point

    for (size_t i = 0; i < PopulationSize / 2; i++) {
      auto &C1 = NewChromosomes[i];
      auto &C2 = NewChromosomes[i + 1];

      int index = Distribution(Engine);

      for (int j = 0; j < index; j++) {
        Chromosomes[i][j] = C1[j];
        Chromosomes[i + 1][j] = C2[j];
      }

      for (size_t j = index; j < ChromosomeSize; j++) {
        Chromosomes[i][j] = C2[j];
        Chromosomes[i + 1][j] = C1[j];
      }
    }
  }

  void mutateGenes() {
    std::random_device RandomDevice;
    std::mt19937 Engine{RandomDevice()};
    std::uniform_int_distribution<unsigned> GeneIndexDistribution{0, static_cast<unsigned>(PopulationSize * ChromosomeSize) - 1};
    std::uniform_int_distribution<unsigned> GenePoolDistribution{0, static_cast<unsigned>(GenePoolSize) - 1};

    int NumMutations = MutationRate * PopulationSize * ChromosomeSize;
    for (int i = 0; i < NumMutations; i++) {
      int GeneIndex = GeneIndexDistribution(Engine);
      Chromosomes[GeneIndex / ChromosomeSize][GeneIndex % ChromosomeSize] = GenePoolDistribution(Engine);
    }
  }
};

std::vector<JITPass> OptimizationPasses;

void InitOptimizationPasses() {
  // 109 is the number of passes in PassRegistry.def
  OptimizationPasses.reserve(109);
  #define MODULE_PASS(NAME, CREATE_PASS) \
    OptimizationPasses.emplace_back(NAME, ModulePass); \

  #define CGSCC_PASS(NAME, CREATE_PASS) \
    OptimizationPasses.emplace_back(NAME, CGSCCPass); \

  #define FUNCTION_PASS(NAME, CREATE_PASS) \
    OptimizationPasses.emplace_back(NAME, FunctionPass); \

  #define LOOP_PASS(NAME, CREATE_PASS) \
    OptimizationPasses.emplace_back(NAME, LoopPass); \

  #include "PassRegistry.def"

  #undef MODULE_PASS
  #undef CGSCC_PASS
  #undef FUNCTION_PASS
  #undef LOOP_PASS
}

bool InitializedDevTarget = false;

struct DevFileData {
  const char *Filename;
  const void *Data;
  size_t DataSize;
};

struct DevData {
  const char *Triple;
  const char *Arch;
  const char *ASTBuffer;
  size_t ASTBufferSize;
  const void *CmdArgs;
  size_t CmdArgsLen;
  DevFileData *FileData;
  size_t FileDataCnt;
};

struct CompilerData {
  std::unique_ptr<CompilerInvocation>     Invocation;
  std::unique_ptr<llvm::opt::OptTable>    Opts;
  IntrusiveRefCntPtr<DiagnosticOptions>   DiagOpts;
  std::unique_ptr<TextDiagnosticPrinter>  DiagnosticPrinter;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  IntrusiveRefCntPtr<DiagnosticsEngine>   Diagnostics;
  IntrusiveRefCntPtr<FileManager>         FileMgr;
  IntrusiveRefCntPtr<SourceManager>       SourceMgr;
  IntrusiveRefCntPtr<MemoryBufferCache>   PCMCache;
  std::unique_ptr<HeaderSearch>           HeaderInfo;
  std::unique_ptr<PCHContainerReader>     PCHContainerRdr;
  IntrusiveRefCntPtr<TargetInfo>          Target;
  std::shared_ptr<Preprocessor>           PP;
  IntrusiveRefCntPtr<ASTContext>          Ctx;
  std::shared_ptr<clang::TargetOptions>   TargetOpts;
  std::shared_ptr<HeaderSearchOptions>    HSOpts;
  std::shared_ptr<PreprocessorOptions>    PPOpts;
  IntrusiveRefCntPtr<ASTReader>           Reader;
  std::shared_ptr<BackendConsumer>        Consumer;
  std::unique_ptr<Sema>                   S;
  TrivialModuleLoader                     ModuleLoader;
  std::unique_ptr<llvm::Module>           RunningMod;

  DenseMap<StringRef, const void *>       LocalSymAddrs;
  DenseMap<StringRef, ValueDecl *>        NewLocalSymDecls;
  std::unique_ptr<ClangJIT>               CJ;

  DenseMap<unsigned, FunctionDecl *>      FuncMap;

  // A map of each instantiation to the containing function. These might not be
  // unique, but should be unique for any place where it matters
  // (instantiations with from-string types).
  DenseMap<unsigned, FunctionDecl *>      CSFuncMap;

  std::unique_ptr<CompilerData>           DevCD;
  SmallString<1>                          DevAsm;
  std::vector<std::unique_ptr<llvm::Module>> DevLinkMods;

  CompilerData(const void *CmdArgs, unsigned CmdArgsLen,
               const void *ASTBuffer, size_t ASTBufferSize,
               const void *IRBuffer, size_t IRBufferSize,
               const void **LocalPtrs, unsigned LocalPtrsCnt,
               const void **LocalDbgPtrs, unsigned LocalDbgPtrsCnt,
               const DevData *DeviceData, unsigned DevCnt,
               int ForDev = -1) {
    bool IsForDev = (ForDev != -1);

    StringRef CombinedArgv((const char *) CmdArgs, CmdArgsLen);
    SmallVector<StringRef, 32> Argv;
    CombinedArgv.split(Argv, '\0', /*MaxSplit*/ -1, false);

    llvm::opt::ArgStringList CC1Args;
    for (auto &ArgStr : Argv)
      CC1Args.push_back(ArgStr.begin());

    unsigned MissingArgIndex, MissingArgCount;
    Opts = driver::createDriverOptTable();
    llvm::opt::InputArgList ParsedArgs = Opts->ParseArgs(
      CC1Args, MissingArgIndex, MissingArgCount);

    DiagOpts = new DiagnosticOptions();
    ParseDiagnosticArgs(*DiagOpts, ParsedArgs);
    DiagnosticPrinter.reset(new TextDiagnosticPrinter(
      llvm::errs(), &*DiagOpts));
    Diagnostics = new DiagnosticsEngine(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      DiagnosticPrinter.get(), false);

    // Note that LangOpts, TargetOpts can also be read from the AST, but
    // CodeGenOpts need to come from the stored command line.

    Invocation.reset(new CompilerInvocation);
    CompilerInvocation::CreateFromArgs(*Invocation,
                                 const_cast<const char **>(CC1Args.data()),
                                 const_cast<const char **>(CC1Args.data()) +
                                 CC1Args.size(), *Diagnostics);
    Invocation->getFrontendOpts().DisableFree = false;
    Invocation->getCodeGenOpts().DisableFree = false;
    Invocation->getCodeGenOpts().ExperimentalNewPassManager = 1;

    InMemoryFileSystem = new llvm::vfs::InMemoryFileSystem;
    FileMgr = new FileManager(FileSystemOptions(), InMemoryFileSystem);

    const char *Filename = "__clang_jit.pcm";
    StringRef ASTBufferSR((const char *) ASTBuffer, ASTBufferSize);
    InMemoryFileSystem->addFile(Filename, 0,
                                llvm::MemoryBuffer::getMemBufferCopy(ASTBufferSR));

    PCHContainerRdr.reset(new RawPCHContainerReader);
    SourceMgr = new SourceManager(*Diagnostics, *FileMgr,
                                  /*UserFilesAreVolatile*/ false);
    PCMCache = new MemoryBufferCache;
    HSOpts = std::make_shared<HeaderSearchOptions>();
    HSOpts->ModuleFormat = PCHContainerRdr->getFormat();
    HeaderInfo.reset(new HeaderSearch(HSOpts,
                                      *SourceMgr,
                                      *Diagnostics,
                                      *Invocation->getLangOpts(),
                                      /*Target=*/nullptr));
    PPOpts = std::make_shared<PreprocessorOptions>();

    unsigned Counter;

    PP = std::make_shared<Preprocessor>(
        PPOpts, *Diagnostics, *Invocation->getLangOpts(),
        *SourceMgr, *PCMCache, *HeaderInfo, ModuleLoader,
        /*IILookup=*/nullptr,
        /*OwnsHeaderSearch=*/false);

    // For parsing type names in strings later, we'll need to have Preprocessor
    // keep the Lexer around even after it hits the end of the each file (used
    // for each type name).
    PP->enableIncrementalProcessing();

    Ctx = new ASTContext(*Invocation->getLangOpts(), *SourceMgr,
                         PP->getIdentifierTable(), PP->getSelectorTable(),
                         PP->getBuiltinInfo());

    Reader = new ASTReader(*PP, Ctx.get(), *PCHContainerRdr, {},
                           /*isysroot=*/"",
                           /*DisableValidation=*/ false,
                           /*AllowPCHWithCompilerErrors*/ false);

    Reader->setListener(llvm::make_unique<ASTInfoCollector>(
      *PP, Ctx.get(), *HSOpts, *PPOpts, *Invocation->getLangOpts(),
      TargetOpts, Target, Counter));

    Ctx->setExternalSource(Reader);

    switch (Reader->ReadAST(Filename, serialization::MK_MainFile,
                            SourceLocation(), ASTReader::ARR_None)) {
    case ASTReader::Success:
      break;

    case ASTReader::Failure:
    case ASTReader::Missing:
    case ASTReader::OutOfDate:
    case ASTReader::VersionMismatch:
    case ASTReader::ConfigurationMismatch:
    case ASTReader::HadErrors:
      Diagnostics->Report(diag::err_fe_unable_to_load_pch);
      fatal();
      return;
    }

    PP->setCounterValue(Counter);

    // Now that we've read the language options from the AST file, change the JIT mode.
    Invocation->getLangOpts()->setCPlusPlusJIT(LangOptions::JITMode::JM_IsJIT);

    // Keep externally available functions, etc.
    Invocation->getCodeGenOpts().PrepareForLTO = true;

    BackendAction BA = Backend_EmitNothing;
    std::unique_ptr<raw_pwrite_stream> OS(new llvm::raw_null_ostream);

    if (IsForDev) {
       BA = Backend_EmitAssembly;
       OS.reset(new raw_svector_ostream(DevAsm));
    }

    Consumer.reset(new BackendConsumer(
        BA, *Diagnostics, Invocation->getHeaderSearchOpts(),
        Invocation->getPreprocessorOpts(), Invocation->getCodeGenOpts(),
        Invocation->getTargetOpts(), *Invocation->getLangOpts(), false, Filename,
        std::move(OS), *LCtx, DevLinkMods));

    // Create a semantic analysis object and tell the AST reader about it.
    S.reset(new Sema(*PP, *Ctx, *Consumer));
    S->Initialize();
    Reader->InitializeSema(*S);

    // Tell the diagnostic client that we have started a source file.
    Diagnostics->getClient()->BeginSourceFile(PP->getLangOpts(), PP.get());

    JFIMapDeclVisitor(FuncMap).TraverseAST(*Ctx);
    JFICSMapDeclVisitor(CSFuncMap).TraverseAST(*Ctx);

    if (IRBufferSize) {
      llvm::SMDiagnostic Err;
      StringRef IRBufferSR((const char *) IRBuffer, IRBufferSize);
      RunningMod = parseIR(
        *llvm::MemoryBuffer::getMemBufferCopy(IRBufferSR), Err, *LCtx);

      for (auto &F : RunningMod->functions())
        if (!F.isDeclaration())
          F.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);

      for (auto &GV : RunningMod->global_values())
        if (!GV.isDeclaration()) {
          if (GV.hasAppendingLinkage())
            cast<GlobalVariable>(GV).setInitializer(nullptr);
          else if (isa<GlobalAlias>(GV))
            // Aliases cannot have externally-available linkage, so give them
            // private linkage.
            GV.setLinkage(llvm::GlobalValue::PrivateLinkage);
          else
            GV.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
        }
    }

    Consumer->Initialize(*Ctx);

    for (unsigned Idx = 0; Idx < 2*LocalPtrsCnt; Idx += 2) {
      const char *Name = (const char *) LocalPtrs[Idx];
      const void *Ptr = LocalPtrs[Idx+1];
      LocalSymAddrs[Name] = Ptr;
    }

    for (unsigned Idx = 0; Idx < 2*LocalDbgPtrsCnt; Idx += 2) {
      const char *Name = (const char *) LocalDbgPtrs[Idx];
      const void *Ptr = LocalDbgPtrs[Idx+1];
      LocalSymAddrs[Name] = Ptr;
    }

    if (!IsForDev)
      CJ = llvm::make_unique<ClangJIT>(LocalSymAddrs);

    if (IsForDev)
      for (unsigned i = 0; i < DeviceData[ForDev].FileDataCnt; ++i) {
        StringRef FileBufferSR(
                    (const char *) DeviceData[ForDev].FileData[i].Data,
                    DeviceData[ForDev].FileData[i].DataSize);

        llvm::SMDiagnostic Err;
        DevLinkMods.push_back(parseIR(
          *llvm::MemoryBuffer::getMemBufferCopy(FileBufferSR), Err, *LCtx));
      }

    if (!IsForDev && Invocation->getLangOpts()->CUDA) {
      typedef int (*cudaGetDevicePtr)(int *);
      auto cudaGetDevice =
        (cudaGetDevicePtr) RTDyldMemoryManager::getSymbolAddressInProcess(
                                                     "cudaGetDevice");
      if (!cudaGetDevice) {
        llvm::errs() << "Could not find CUDA API functions; "
                        "did you forget to link with -lcudart?\n";
        fatal();
      }

      typedef int (*cudaGetDeviceCountPtr)(int *);
      auto cudaGetDeviceCount =
        (cudaGetDeviceCountPtr) RTDyldMemoryManager::getSymbolAddressInProcess(
                                                     "cudaGetDeviceCount");

      int SysDevCnt;
      if (cudaGetDeviceCount(&SysDevCnt)) {
        llvm::errs() << "Failed to get CUDA device count!\n";
        fatal();
      }

      typedef int (*cudaDeviceGetAttributePtr)(int *, int, int);
      auto cudaDeviceGetAttribute =
        (cudaDeviceGetAttributePtr) RTDyldMemoryManager::getSymbolAddressInProcess(
                                      "cudaDeviceGetAttribute");

      if (SysDevCnt) {
        int CDev;
        if (cudaGetDevice(&CDev))
          fatal();

        int CLMajor, CLMinor;
        if (cudaDeviceGetAttribute(
              &CLMajor, /*cudaDevAttrComputeCapabilityMajor*/ 75, CDev))
          fatal();
        if (cudaDeviceGetAttribute(
              &CLMinor, /*cudaDevAttrComputeCapabilityMinor*/ 76, CDev))
          fatal();

        SmallString<6> EffArch;
        raw_svector_ostream(EffArch) << "sm_" << CLMajor << CLMinor;

        SmallVector<StringRef, 2> DevArchs;
        for (unsigned i = 0; i < DevCnt; ++i) {
          if (!Triple(DeviceData[i].Triple).isNVPTX())
            continue;
          if (!StringRef(DeviceData[i].Arch).startswith("sm_"))
            continue;
          DevArchs.push_back(DeviceData[i].Arch);
        }

        std::sort(DevArchs.begin(), DevArchs.end());
        auto ArchI =
          std::upper_bound(DevArchs.begin(), DevArchs.end(), EffArch);
        if (ArchI == DevArchs.begin()) {
          llvm::errs() << "No JIT device configuration supports " <<
                          EffArch << "\n";
          fatal();
        }

        auto BestDevArch = *--ArchI;
        int BestDevIdx = 0;
        for (; BestDevIdx < (int) DevCnt; ++BestDevIdx) {
          if (!Triple(DeviceData[BestDevIdx].Triple).isNVPTX())
            continue;
          if (DeviceData[BestDevIdx].Arch == BestDevArch)
            break;
        }

        assert(BestDevIdx != (int) DevCnt && "Didn't find the chosen device data?");

        if (!InitializedDevTarget) {
          // In theory, we only need to initialize the NVPTX target here,
          // however, there doesn't seem to be any good way to know if the
          // NVPTX target is enabled.
          //
          // LLVMInitializeNVPTXTargetInfo();
          // LLVMInitializeNVPTXTarget();
          // LLVMInitializeNVPTXTargetMC();
          // LLVMInitializeNVPTXAsmPrinter();

          llvm::InitializeAllTargets();
          llvm::InitializeAllTargetMCs();
          llvm::InitializeAllAsmPrinters();

          InitializedDevTarget = true;
        }

        DevCD.reset(new CompilerData(
            DeviceData[BestDevIdx].CmdArgs, DeviceData[BestDevIdx].CmdArgsLen,
            DeviceData[BestDevIdx].ASTBuffer, DeviceData[BestDevIdx].ASTBufferSize,
            nullptr, 0, nullptr, 0, nullptr, 0, DeviceData, DevCnt, BestDevIdx));
      }
    }

    // auto FileName = SourceMgr->getFileEntryForID(SourceMgr->getMainFileID())->getName();
    // llvm::errs() << "got name\n";
    // llvm::errs() << "JIT: file ID " << FileName << '\n';

    // llvm::errs() << "JIT: Translation unit decl\n";
    // Ctx->getTranslationUnitDecl()->dump();

    if (Invocation->getFrontendOpts().ShowStats || !Invocation->getFrontendOpts().StatsFile.empty())
      llvm::EnableStatistics(false);
  }

  std::string joinPasses(const SmallVectorImpl<StringRef> &Passes, StringRef Prefix, int NumParens) {
    if (Passes.size() == 0)
      return "";

    return formatv("{0}{1:$[,]}{2}", Prefix, make_range(Passes.begin(), Passes.end()), fmt_repeat(")", NumParens));
  }

  std::string buildPassPipeline(int VariantIdx) {
    JITPipeline Pipeline;

    if (VariantIdx == 3)
      Pipeline.buildPassPipeline(PassBuilder::O1);

    if (VariantIdx == 4)
      Pipeline.buildPassPipeline(PassBuilder::O2);

    if (VariantIdx == 5)
      Pipeline.buildPassPipeline(PassBuilder::O3);

    return Pipeline.toString();
  }

  void restoreFuncDeclContext(FunctionDecl *FunD) {
    // NOTE: This mirrors the corresponding code in
    // Parser::ParseLateTemplatedFuncDef (which is used to late parse a C++
    // function template in Microsoft mode).

    struct ContainingDC {
      ContainingDC(DeclContext *DC, bool ShouldPush) : Pair(DC, ShouldPush) {}
      llvm::PointerIntPair<DeclContext *, 1, bool> Pair;
      DeclContext *getDC() { return Pair.getPointer(); }
      bool shouldPushDC() { return Pair.getInt(); }
    };

    SmallVector<ContainingDC, 4> DeclContextsToReenter;
    DeclContext *DD = FunD;
    DeclContext *NextContaining = S->getContainingDC(DD);
    while (DD && !DD->isTranslationUnit()) {
      bool ShouldPush = DD == NextContaining;
      DeclContextsToReenter.push_back({DD, ShouldPush});
      if (ShouldPush)
        NextContaining = S->getContainingDC(DD);
      DD = DD->getLexicalParent();
    }

    // Reenter template scopes from outermost to innermost.
    for (ContainingDC CDC : reverse(DeclContextsToReenter)) {
      (void) S->ActOnReenterTemplateScope(S->getCurScope(),
                                           cast<Decl>(CDC.getDC()));
      if (CDC.shouldPushDC())
        S->PushDeclContext(S->getCurScope(), CDC.getDC());
    }
  }

  std::string instantiateTemplate(const void *NTTPValues, const char **TypeStrings,
                                  unsigned Idx, unsigned &VariantIdx) {
    FunctionDecl *FD = FuncMap[Idx];
    if (!FD)
      fatal();

    RecordDecl *RD =
      Ctx->buildImplicitRecord(llvm::Twine("__clang_jit_args_")
                               .concat(llvm::Twine(Idx))
                               .concat(llvm::Twine("_t"))
                               .str());

    RD->startDefinition();

    enum TASaveKind {
      TASK_None,
      TASK_Type,
      TASK_Value
    };

    SmallVector<TASaveKind, 8> TAIsSaved;

    auto *FTSI = FD->getTemplateSpecializationInfo();
    for (auto &TA : FTSI->TemplateArguments->asArray()) {
      auto HandleTA = [&](const TemplateArgument &TA) {
        if (TA.getKind() == TemplateArgument::Type)
          if (TA.getAsType()->isJITFromStringType()) {
            TAIsSaved.push_back(TASK_Type);
            return;
          }

        if (TA.getKind() != TemplateArgument::Expression) {
          TAIsSaved.push_back(TASK_None);
          return;
        }

        SmallVector<PartialDiagnosticAt, 8> Notes;
        Expr::EvalResult Eval;
        Eval.Diag = &Notes;
        if (TA.getAsExpr()->
              EvaluateAsConstantExpr(Eval, Expr::EvaluateForMangling, *Ctx)) {
          TAIsSaved.push_back(TASK_None);
          return;
        }

        QualType FieldTy = TA.getNonTypeTemplateArgumentType();
        auto *Field = FieldDecl::Create(
            *Ctx, RD, SourceLocation(), SourceLocation(), /*Id=*/nullptr,
            FieldTy, Ctx->getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
            /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
        Field->setAccess(AS_public);
        RD->addDecl(Field);

        TAIsSaved.push_back(TASK_Value);
      };

      if (TA.getKind() == TemplateArgument::Pack) {
        for (auto &PTA : TA.getPackAsArray())
          HandleTA(PTA);
        continue;
      }

      HandleTA(TA);
    }

    RD->completeDefinition();
    RD->addAttr(PackedAttr::CreateImplicit(*Ctx));

    const ASTRecordLayout &RLayout = Ctx->getASTRecordLayout(RD);
    assert(Ctx->getCharWidth() == 8 && "char is not 8 bits!");

    QualType RDTy = Ctx->getRecordType(RD);
    auto Fields = cast<RecordDecl>(RDTy->getAsTagDecl())->field_begin();

    SmallVector<TemplateArgument, 8> Builder;

    unsigned TAIdx = 0, TSIdx = 0;
    for (auto &TA : FTSI->TemplateArguments->asArray()) {
      auto HandleTA = [&](const TemplateArgument &TA,
                          SmallVector<TemplateArgument, 8> &Builder) {
        if (TAIsSaved[TAIdx] == TASK_Type) {
          PP->ResetForJITTypes();

          PP->setPredefines(TypeStrings[TSIdx]);
          PP->EnterMainSourceFile();

          Parser P(*PP, *S, /*SkipFunctionBodies*/true, /*JITTypes*/true);

          // Reset this to nullptr so that when we call
          // Parser::Initialize it has the clean slate it expects.
          S->CurContext = nullptr;

          P.Initialize();

          Sema::ContextRAII TUContext(*S, Ctx->getTranslationUnitDecl());

          auto CSFMI = CSFuncMap.find(Idx);
          if (CSFMI != CSFuncMap.end()) {
	  // Note that this restores the context of the function in which the
	  // template was instantiated, but not the state *within* the
	  // function, so local types will remain unavailable.

            auto *FunD = CSFMI->second;
            restoreFuncDeclContext(FunD);
            S->CurContext = S->getContainingDC(FunD);
          }

          TypeResult TSTy = P.ParseTypeName();
          if (TSTy.isInvalid())
            fatal();

          QualType TypeFromString = Sema::GetTypeFromParser(TSTy.get());
          TypeFromString = Ctx->getCanonicalType(TypeFromString);

          Builder.push_back(TemplateArgument(TypeFromString));

          ++TSIdx;
          ++TAIdx;
          return;
        }

        if (TAIsSaved[TAIdx++] != TASK_Value) {
          Builder.push_back(TA);
          VariantIdx = TA.getAsIntegral().getExtValue();
          return;
        }

        assert(TA.getKind() == TemplateArgument::Expression &&
               "Only expressions template arguments handled here");

        QualType FieldTy = TA.getNonTypeTemplateArgumentType();

        assert(!FieldTy->isMemberPointerType() &&
               "Can't handle member pointers here without ABI knowledge");

        auto *Fld = *Fields++;
        unsigned Offset = RLayout.getFieldOffset(Fld->getFieldIndex()) / 8;
        unsigned Size = Ctx->getTypeSizeInChars(FieldTy).getQuantity();

        unsigned NumIntWords = llvm::alignTo<8>(Size);
        SmallVector<uint64_t, 2> IntWords(NumIntWords, 0);
        std::memcpy((char *) IntWords.data(),
                    ((const char *) NTTPValues) + Offset, Size);
        llvm::APInt IntVal(Size*8, IntWords);

        QualType CanonFieldTy = Ctx->getCanonicalType(FieldTy);

        if (FieldTy->isIntegralOrEnumerationType()) {
          llvm::APSInt SIntVal(IntVal,
                               FieldTy->isUnsignedIntegerOrEnumerationType());
          Builder.push_back(TemplateArgument(*Ctx, SIntVal, CanonFieldTy));
          VariantIdx = SIntVal.getExtValue();
        } else {
          assert(FieldTy->isPointerType() || FieldTy->isReferenceType() ||
                 FieldTy->isNullPtrType());
          if (IntVal.isNullValue()) {
            Builder.push_back(TemplateArgument(CanonFieldTy, /*isNullPtr*/true));
          } else {
	  // Note: We always generate a new global for pointer values here.
	  // This provides a new potential way to introduce an ODR violation:
	  // If you also generate an instantiation using the same pointer value
	  // using some other symbol name, this will generate a different
	  // instantiation.

	  // As we guarantee that the template parameters are not allowed to
	  // point to subobjects, this is useful for optimization because each
	  // of these resolve to distinct underlying objects.

            llvm::SmallString<256> GlobalName("__clang_jit_symbol_");
            IntVal.toString(GlobalName, 16, false);

	  // To this base name we add the mangled type. Stack/heap addresses
	  // can be reused with variables of different type, and these should
	  // have different names even if they share the same address;
            auto &CGM = Consumer->getCodeGenerator()->CGM();
            llvm::raw_svector_ostream MOut(GlobalName);
            CGM.getCXXABI().getMangleContext().mangleTypeName(CanonFieldTy, MOut);

            auto NLDSI = NewLocalSymDecls.find(GlobalName);
            if (NLDSI != NewLocalSymDecls.end()) {
                Builder.push_back(TemplateArgument(NLDSI->second, CanonFieldTy));
            } else {
              Sema::ContextRAII TUContext(*S, Ctx->getTranslationUnitDecl());
              SourceLocation Loc = FTSI->getPointOfInstantiation();

              QualType STy = CanonFieldTy->getPointeeType();
              auto &II = PP->getIdentifierTable().get(GlobalName);

              if (STy->isFunctionType()) {
                auto *TAFD =
                  FunctionDecl::Create(*Ctx, S->CurContext, Loc, Loc, &II,
                                       STy, /*TInfo=*/nullptr, SC_Extern, false,
                                       STy->isFunctionProtoType());
                TAFD->setImplicit();

                if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(STy)) {
                  SmallVector<ParmVarDecl*, 16> Params;
                  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
                    ParmVarDecl *Parm =
                      ParmVarDecl::Create(*Ctx, TAFD, SourceLocation(), SourceLocation(),
                                          nullptr, FT->getParamType(i), /*TInfo=*/nullptr,
                                          SC_None, nullptr);
                    Parm->setScopeInfo(0, i);
                    Params.push_back(Parm);
                  }

                  TAFD->setParams(Params);
                }

                NewLocalSymDecls[II.getName()] = TAFD;
                Builder.push_back(TemplateArgument(TAFD, CanonFieldTy));
              } else {
                bool MadeArray = false;
                auto *TPL = FTSI->getTemplate()->getTemplateParameters();
                if (TPL->size() >= TAIdx) {
                  auto *Param = TPL->getParam(TAIdx-1);
                  if (NonTypeTemplateParmDecl *NTTP =
                        dyn_cast<NonTypeTemplateParmDecl>(Param)) {
                    QualType OrigTy = NTTP->getType()->getPointeeType();
                    OrigTy = OrigTy.getDesugaredType(*Ctx);

                    bool IsArray = false;
                    llvm::APInt Sz;
                    QualType ElemTy;
                    if (const auto *DAT = dyn_cast<DependentSizedArrayType>(OrigTy)) {
                      Expr* SzExpr = DAT->getSizeExpr();

                      // Get the already-processed arguments for potential substitution.
                      auto *NewTAL = TemplateArgumentList::CreateCopy(*Ctx, Builder);
                      MultiLevelTemplateArgumentList SubstArgs(*NewTAL);

                      SmallVector<Expr *, 1> NewSzExprVec;
                      if (!S->SubstExprs(SzExpr, /*IsCall*/ false, SubstArgs, NewSzExprVec)) {
                        Expr::EvalResult NewSzResult;
                        if (NewSzExprVec[0]->EvaluateAsInt(NewSzResult, *Ctx)) {
                          Sz = NewSzResult.Val.getInt();
                          ElemTy = DAT->getElementType();
                          IsArray = true;
                        }
                      }
                    } else if (const auto *CAT = dyn_cast<ConstantArrayType>(OrigTy)) {
                      Sz = CAT->getSize();
                      ElemTy = CAT->getElementType();
                      IsArray = true;
                    }

                    if (IsArray && (ElemTy->isIntegerType() ||
                                    ElemTy->isFloatingType())) {
                      QualType ArrTy =
                        Ctx->getConstantArrayType(ElemTy,
                                                  Sz, clang::ArrayType::Normal, 0);

                      SmallVector<Expr *, 16> Vals;
                      unsigned ElemSize = Ctx->getTypeSizeInChars(ElemTy).getQuantity();
                      unsigned ElemNumIntWords = llvm::alignTo<8>(ElemSize);
                      const char *Elem = (const char *) IntVal.getZExtValue();
                      for (unsigned i = 0; i < Sz.getZExtValue(); ++i) {
                        SmallVector<uint64_t, 2> ElemIntWords(ElemNumIntWords, 0);

                        std::memcpy((char *) ElemIntWords.data(), Elem, ElemSize);
                        Elem += ElemSize;

                        llvm::APInt ElemVal(ElemSize*8, ElemIntWords);
                        if (ElemTy->isIntegerType()) {
                          Vals.push_back(new (*Ctx) IntegerLiteral(
                            *Ctx, ElemVal, ElemTy, Loc));
                        } else {
                          llvm::APFloat ElemValFlt(Ctx->getFloatTypeSemantics(ElemTy), ElemVal);
                          Vals.push_back(FloatingLiteral::Create(*Ctx, ElemValFlt,
                                                                 false, ElemTy, Loc));
                        }
                      }

                      InitListExpr *InitL = new (*Ctx) InitListExpr(*Ctx, Loc, Vals, Loc);
                      InitL->setType(ArrTy);

                      auto *TAVD =
                        VarDecl::Create(*Ctx, S->CurContext, Loc, Loc, &II,
                                        ArrTy, Ctx->getTrivialTypeSourceInfo(ArrTy, Loc),
                                        SC_Extern);
                      TAVD->setImplicit();
                      TAVD->setConstexpr(true);
                      TAVD->setInit(InitL);

                      NewLocalSymDecls[II.getName()] = TAVD;
                      Builder.push_back(TemplateArgument(TAVD, Ctx->getLValueReferenceType(ArrTy)));

                      MadeArray = true;
                    }
                  }
                }

                if (!MadeArray) {
                  auto *TAVD =
                    VarDecl::Create(*Ctx, S->CurContext, Loc, Loc, &II,
                                    STy, Ctx->getTrivialTypeSourceInfo(STy, Loc),
                                    SC_Extern);
                  TAVD->setImplicit();

                  NewLocalSymDecls[II.getName()] = TAVD;
                  Builder.push_back(TemplateArgument(TAVD, CanonFieldTy));
                }
              }

              LocalSymAddrs[II.getName()] = (const void *) IntVal.getZExtValue();
            }
          }
        }
      };

      if (TA.getKind() == TemplateArgument::Pack) {
        SmallVector<TemplateArgument, 8> PBuilder;
        for (auto &PTA : TA.getPackAsArray())
          HandleTA(PTA, PBuilder);
        Builder.push_back(TemplateArgument::CreatePackCopy(*Ctx, PBuilder));
        continue;
      }

      HandleTA(TA, Builder);
    }

    llvm::errs() << "JIT: selecting variant index " << VariantIdx << '\n';

    SourceLocation Loc = FTSI->getPointOfInstantiation();
    auto *NewTAL = TemplateArgumentList::CreateCopy(*Ctx, Builder);
    MultiLevelTemplateArgumentList SubstArgs(*NewTAL);

    auto *FunctionTemplate = FTSI->getTemplate();
    DeclContext *Owner = FunctionTemplate->getDeclContext();
    if (FunctionTemplate->getFriendObjectKind())
      Owner = FunctionTemplate->getLexicalDeclContext();

    std::string SMName;
    FunctionTemplateDecl *FTD = FTSI->getTemplate();
    sema::TemplateDeductionInfo Info(Loc);
    {
      Sema::InstantiatingTemplate Inst(
        *S, Loc, FTD, NewTAL->asArray(),
        Sema::CodeSynthesisContext::ExplicitTemplateArgumentSubstitution, Info);

      S->setCurScope(S->TUScope = new Scope(nullptr, Scope::DeclScope, PP->getDiagnostics()));

      Sema::ContextRAII TUContext(*S, Ctx->getTranslationUnitDecl());

      auto *Specialization = cast_or_null<FunctionDecl>(
        S->SubstDecl(FunctionTemplate->getTemplatedDecl(), Owner, SubstArgs));
      if (!Specialization || Specialization->isInvalidDecl())
        fatal();

      Specialization->setTemplateSpecializationKind(TSK_ExplicitInstantiationDefinition, Loc);
      S->InstantiateFunctionDefinition(Loc, Specialization, true, true, true);

      SMName = Consumer->getCodeGenerator()->CGM().getMangledName(Specialization);
    }

    if (Diagnostics->hasErrorOccurred())
      fatal();

    return SMName;
  }

  void emitAllNeeded(bool CheckExisting = true) {
    // There might have been functions/variables with local linkage that were
    // only used by JIT functions. These would not have been used during
    // initial code generation for this translation unit, and so not emitted.
    // We need to make sure that they're emited now (if they're now necessary).

    // Note that we skip having the code generator visiting the decl if it is
    // already defined or already present in our running module. Note that this
    // is not sufficient to prevent all redundant code generation (this might
    // also happen during the instantiation of the top-level function
    // template), and this is why we merge the running module into the new one
    // with the running-module overriding new entities.

    SmallSet<StringRef, 16> LastDeclNames;
    bool Changed;
    do {
      Changed = false;

      Consumer->getCodeGenerator()->CGM().EmitAllDeferred([&](GlobalDecl GD) {
        auto MName = Consumer->getCodeGenerator()->CGM().getMangledName(GD);
        if (!CheckExisting || !CJ->findSymbol(MName)) {
          Changed = true;
          return false;
        }

        return true;
      });

      SmallSet<StringRef, 16> DeclNames;
      for (auto &F : Consumer->getModule()->functions())
        if (F.isDeclaration() && !F.isIntrinsic())
          if (!LastDeclNames.count(F.getName()))
            DeclNames.insert(F.getName());

      for (auto &GV : Consumer->getModule()->global_values())
        if (GV.isDeclaration())
          if (!LastDeclNames.count(GV.getName()))
            DeclNames.insert(GV.getName());

      for (auto &DeclName : DeclNames) {
        if (CheckExisting && CJ->findSymbol(DeclName))
          continue;

        Decl *D = const_cast<Decl *>(Consumer->getCodeGenerator()->
                                       GetDeclForMangledName(DeclName));
        if (!D)
          continue;

        Consumer->HandleInterestingDecl(DeclGroupRef(D));
        LastDeclNames.insert(DeclName);
        Changed = true;
      }
    } while (Changed);
  }

  void *resolveFunction(const void *NTTPValues, const char **TypeStrings,
                        unsigned Idx) {
    unsigned VariantIdx = 0;
    std::string SMName = instantiateTemplate(NTTPValues, TypeStrings, Idx, VariantIdx);

    llvm::errs() << "JIT: searching for variant, Index: " << Idx << "; Name: " << SMName << " ...\n";

    // Now we know the name of the symbol, check to see if we already have it.
    if (auto SpecSymbol = CJ->findSymbol(SMName))
      if (SpecSymbol.getAddress())
        return (void *) llvm::cantFail(SpecSymbol.getAddress());

    llvm::errs() << "JIT: not found, re-compiling\n";

    if (DevCD)
      DevCD->instantiateTemplate(NTTPValues, TypeStrings, Idx, VariantIdx);

    emitAllNeeded();

    if (DevCD)
      DevCD->emitAllNeeded(false);

    // Before anything gets optimized, mark the top-level symbol we're
    // generating so that it doesn't get eliminated by the optimizer.

    auto *TopGV =
      cast<GlobalObject>(Consumer->getModule()->getNamedValue(SMName));
    assert(TopGV && "Didn't generate the desired top-level symbol?");

    TopGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
    TopGV->setComdat(nullptr);

    std::string PassPipeline = "";
    // This is needed to pass the correct argument to the inlining pass
    Invocation->getCodeGenOpts().OptimizationLevel = 1;

    if (VariantIdx < 3)
      Invocation->getCodeGenOpts().OptimizationLevel = VariantIdx + 1;
    else if (VariantIdx == 3) { // Genetic search
      JITPipeline JP;
      JP.buildPassPipeline(PassBuilder::OptimizationLevel::O3);

      GeneticPipelineBuilder GPB(Consumer, 25, JP.Passes.size(), JP.Passes.size(), std::move(JP.Passes));
      PassPipeline = GPB.Run();
    } else if (VariantIdx == 4) { // Random search
      if (BadPipelines.size() == 0)
        readBadPipelines();

      JITPipeline JP;
      JP.buildPassPipeline(PassBuilder::OptimizationLevel::O3);

      RandomPipelineBuilder RPB(Consumer, 200, JP.Passes.size(), std::move(JP.Passes));
      PassPipeline = RPB.Run();
    } else if (VariantIdx == 5) { // Random search once
      // if (BadPipelines.size() == 0)
      //   readBadPipelines();

      if (CurrentKernel == 0) {
        JITPipeline JP;
        JP.buildPassPipeline(PassBuilder::OptimizationLevel::O3);
        RandomPipelineBuilder RPB(Consumer, 200, JP.Passes.size(), std::move(JP.Passes));
        RandomPipeline = RPB.Run();
      }

      PassPipeline = RandomPipeline;
      llvm::errs() << "Using pipeline: " << PassPipeline << '\n';
    } else if (VariantIdx == 6) { // Save the BC file
      Invocation->getCodeGenOpts().OptimizationLevel = 0;
      std::error_code EC;
      std::string BCFileName = llvm::Twine("out.")
                               .concat(llvm::Twine(Idx))
                               .concat(llvm::Twine(".bc"))
                               .str();

      auto BCFile = llvm::make_unique<llvm::raw_fd_ostream>(BCFileName, EC);
      if (EC) {
        llvm::errs() << "Can't write bitcode to file\n";
      } else {
        llvm::Module* ToWriteMod = Consumer->getModule();
        llvm::WriteBitcodeToFile(*ToWriteMod, *BCFile);
      }
    } else if (VariantIdx == 7) { // Read the pipeline from an environment variable
      PassPipeline = std::string(::getenv("LLVM_JIT_PIPELINE"));
      llvm::errs() << "JIT: got pipeline " << PassPipeline << '\n';
    } else if (VariantIdx == 8) { // Read the pipeline from an environment variable per kernel
      std::string EnvironmentVariable = llvm::Twine("LLVM_JIT_PIPELINE_")
                                        .concat(llvm::Twine(CurrentKernel))
                                        .str();
      auto EnvPipeline = std::string(::getenv(EnvironmentVariable.c_str()));

      if (EnvPipeline == "O3")
        Invocation->getCodeGenOpts().OptimizationLevel = 3;
      else {
        PassPipeline = EnvPipeline;
        llvm::errs() << "JIT: got pipeline " << PassPipeline << '\n';
      }
    } else
      Invocation->getCodeGenOpts().OptimizationLevel = 3;

    // Finalize the module, generate module-level metadata, etc.

    if (DevCD) {
      DevCD->Consumer->HandleTranslationUnit(*DevCD->Ctx);
      DevCD->Consumer->EmitOptimized(PassPipeline);

      // We have now created the PTX output, but what we really need as a
      // fatbin that the CUDA runtime will recognize.

      // The outer header of the fat binary is documented in the CUDA
      // fatbinary.h header. As mentioned there, the overall size must be a
      // multiple of eight, and so we must make sure that the PTX is.
      // We also need to make sure that the buffer is explicitly null
      // terminated (cuobjdump, at least, seems to assume that it is).
      DevCD->DevAsm += '\0';
      while (DevCD->DevAsm.size() % 8)
        DevCD->DevAsm += '\0';

      // NVIDIA, unfortunatly, does not provide full documentation on their
      // fatbin format. There is some information on the outer header block in
      // the CUDA fatbinary.h header. Also, it is possible to figure out more
      // about the format by creating fatbins using the provided utilities
      // and then observing what cuobjdump reports about the resulting files.
      // There are some other online references which shed light on the format,
      // including https://reviews.llvm.org/D8397 and FatBinaryContext.{cpp,h}
      // from the GPU Ocelot project (https://github.com/gtcasl/gpuocelot).

      SmallString<128> FatBin;
      llvm::raw_svector_ostream FBOS(FatBin);

      struct FatBinHeader {
        uint32_t Magic;      // 0x00
        uint16_t Version;    // 0x04
        uint16_t HeaderSize; // 0x06
        uint32_t DataSize;   // 0x08
        uint32_t unknown0c;  // 0x0c
      public:
        FatBinHeader(uint32_t DataSize)
            : Magic(0xba55ed50), Version(1),
              HeaderSize(sizeof(*this)), DataSize(DataSize), unknown0c(0) {}
      };

      enum FatBinFlags {
        AddressSize64 = 0x01,
        HasDebugInfo = 0x02,
        ProducerCuda = 0x04,
        HostLinux = 0x10,
        HostMac = 0x20,
        HostWindows = 0x40
      };

      struct FatBinFileHeader {
        uint16_t Kind;             // 0x00
        uint16_t unknown02;        // 0x02
        uint32_t HeaderSize;       // 0x04
        uint32_t DataSize;         // 0x08
        uint32_t unknown0c;        // 0x0c
        uint32_t CompressedSize;   // 0x10
        uint32_t SubHeaderSize;    // 0x14
        uint16_t VersionMinor;     // 0x18
        uint16_t VersionMajor;     // 0x1a
        uint32_t CudaArch;         // 0x1c
        uint32_t unknown20;        // 0x20
        uint32_t unknown24;        // 0x24
        uint32_t Flags;            // 0x28
        uint32_t unknown2c;        // 0x2c
        uint32_t unknown30;        // 0x30
        uint32_t unknown34;        // 0x34
        uint32_t UncompressedSize; // 0x38
        uint32_t unknown3c;        // 0x3c
        uint32_t unknown40;        // 0x40
        uint32_t unknown44;        // 0x44
        FatBinFileHeader(uint32_t DataSize, uint32_t CudaArch, uint32_t Flags)
            : Kind(1 /*PTX*/), unknown02(0x0101), HeaderSize(sizeof(*this)),
              DataSize(DataSize), unknown0c(0), CompressedSize(0),
              SubHeaderSize(HeaderSize - 8), VersionMinor(2), VersionMajor(4),
              CudaArch(CudaArch), unknown20(0), unknown24(0), Flags(Flags), unknown2c(0),
              unknown30(0), unknown34(0), UncompressedSize(0), unknown3c(0),
              unknown40(0), unknown44(0) {}
      };

      uint32_t CudaArch;
      StringRef(DevCD->Invocation->getTargetOpts().CPU)
        .drop_front(3 /*sm_*/).getAsInteger(10, CudaArch);

      uint32_t Flags = ProducerCuda;
      if (DevCD->Invocation->getCodeGenOpts().getDebugInfo() >=
            codegenoptions::LimitedDebugInfo)
        Flags |= HasDebugInfo;

      if (Triple(DevCD->Invocation->getTargetOpts().Triple).getArch() ==
            Triple::nvptx64)
        Flags |= AddressSize64;

      if (Triple(Invocation->getTargetOpts().Triple).isOSWindows())
        Flags |= HostWindows;
      else if (Triple(Invocation->getTargetOpts().Triple).isOSDarwin())
        Flags |= HostMac;
      else
        Flags |= HostLinux;

      FatBinFileHeader FBFHdr(DevCD->DevAsm.size(), CudaArch, Flags);
      FatBinHeader FBHdr(DevCD->DevAsm.size() + FBFHdr.HeaderSize);

      FBOS.write((char *) &FBHdr, FBHdr.HeaderSize);
      FBOS.write((char *) &FBFHdr, FBFHdr.HeaderSize);
      FBOS << DevCD->DevAsm;

      if (::getenv("CLANG_JIT_CUDA_DUMP_DYNAMIC_FATBIN")) {
        SmallString<128> Path;
        auto EC = llvm::sys::fs::createUniqueFile(
                      llvm::Twine("clang-jit-") +
                      llvm::sys::path::filename(Invocation->getCodeGenOpts().
                                                  MainFileName) +
                      llvm::Twine("-%%%%.fatbin"), Path,
                    llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
        if (!EC) {
          raw_fd_ostream DOS(Path, EC);
          if (!EC)
            DOS << FatBin;
        }
      }

      Consumer->getCodeGenerator()->CGM().getCodeGenOpts().GPUBinForJIT =
        FatBin;
      DevCD->DevAsm.clear();
    }

    // Finalize translation unit. No optimization yet.
    Consumer->HandleTranslationUnit(*Ctx);

    // First, mark everything we've newly generated with external linkage. When
    // we generate additional modules, we'll mark these functions as available
    // externally, and so we're likely to inline them, but if not, we'll need
    // to link with the ones generated here.

    for (auto &F : Consumer->getModule()->functions()) {
      F.setLinkage(llvm::GlobalValue::ExternalLinkage);
      F.setComdat(nullptr);
    }

    auto IsLocalUnnamedConst = [](llvm::GlobalValue &GV) {
      if (!GV.hasAtLeastLocalUnnamedAddr() || !GV.hasLocalLinkage())
        return false;

      auto *GVar = dyn_cast<llvm::GlobalVariable>(&GV);
      if (!GVar || !GVar->isConstant())
        return false;

      return true;
    };

    for (auto &GV : Consumer->getModule()->global_values()) {
      if (IsLocalUnnamedConst(GV) || GV.hasAppendingLinkage())
        continue;

      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      if (auto *GO = dyn_cast<llvm::GlobalObject>(&GV))
        GO->setComdat(nullptr);
    }

    // Here we link our previous cache of definitions, etc. into this module.
    // This includes all of our previously-generated functions (marked as
    // available externally). We prefer our previously-generated versions to
    // our current versions should both modules contain the same entities (as
    // the previously-generated versions have already been optimized).

    // We need to be specifically careful about constants in our module,
    // however. Clang will generate all string literals as .str (plus a
    // number), and these from previously-generated code will conflict with the
    // names chosen for string literals in this module.

    for (auto &GV : Consumer->getModule()->global_values()) {
      if (!IsLocalUnnamedConst(GV) && !GV.getName().startswith("__cuda_") && !GV.getName().startswith(".omp_outlined."))
        continue;

      if (!RunningMod->getNamedValue(GV.getName()))
        continue;

      llvm::SmallString<16> UniqueName(GV.getName());
      unsigned BaseSize = UniqueName.size();
      do {
        // Trim any suffix off and append the next number.
        UniqueName.resize(BaseSize);
        llvm::raw_svector_ostream S(UniqueName);
        S << "." << ++LastUnique;
      } while (RunningMod->getNamedValue(UniqueName));

      GV.setName(UniqueName);
    }

    // Clang will generate local init/deinit functions for variable
    // initialization, CUDA registration, etc. and these can't be shared with
    // the base part of the module (as they specifically initialize variables,
    // etc. that we just generated).

    for (auto &F : Consumer->getModule()->functions()) {
      // FIXME: This likely covers the set of TU-local init/deinit functions
      // that can't be shared with the base module. There should be a better
      // way to do this (e.g., we could record all functions that
      // CreateGlobalInitOrDestructFunction creates? - ___cuda_ would still be
      // a special case).
      if (!F.getName().startswith("__cuda_") &&
          !F.getName().startswith("_GLOBAL_") &&
          !F.getName().startswith("__GLOBAL_") &&
          !F.getName().startswith("__cxx_"))
        continue;

      if (!RunningMod->getFunction(F.getName()))
        continue;

      llvm::SmallString<16> UniqueName(F.getName());
      unsigned BaseSize = UniqueName.size();
      do {
        // Trim any suffix off and append the next number.
        UniqueName.resize(BaseSize);
        llvm::raw_svector_ostream S(UniqueName);
        S << "." << ++LastUnique;
      } while (RunningMod->getFunction(UniqueName));

      F.setName(UniqueName);
    }

    if (Linker::linkModules(*Consumer->getModule(), llvm::CloneModule(*RunningMod),
                            Linker::Flags::OverrideFromSrc))
      fatal(PassPipeline);

    // Aliases are not allowed to point to functions with available_externally linkage.
    // We solve this by replacing these aliases with the definition of the aliasee.
    // Candidates are identified first, then erased in a second step to avoid invalidating the iterator.
    auto& LinkedMod = *Consumer->getModule();
    SmallPtrSet<GlobalAlias*, 4> ToReplace;
    for (auto& Alias : LinkedMod.aliases()) {
      // Aliases may point to other aliases but we only need to alter the lowest level one
      // Only function declarations are relevant
      auto Aliasee = dyn_cast<Function>(Alias.getAliasee());
      if (!Aliasee || !Aliasee->isDeclarationForLinker()) {
        continue;
      }
      assert(Aliasee->hasAvailableExternallyLinkage() && "Broken module: alias points to declaration");
      ToReplace.insert(&Alias);
    }

    for (auto* Alias : ToReplace) {
      auto Aliasee = cast<Function>(Alias->getAliasee());

      llvm::ValueToValueMapTy VMap;
      Function* AliasReplacement = llvm::CloneFunction(Aliasee, VMap);

      AliasReplacement->setLinkage(Alias->getLinkage());
      Alias->replaceAllUsesWith(AliasReplacement);

      SmallString<32> AliasName = Alias->getName();
      Alias->eraseFromParent();
      AliasReplacement->setName(AliasName);
    }

    // Optimize the merged module, containing both the newly generated IR as well as
    // previously emitted code marked available_externally.
    Consumer->EmitOptimized(PassPipeline);

    std::unique_ptr<llvm::Module> ToRunMod =
        llvm::CloneModule(*Consumer->getModule());

    CJ->addModule(std::move(ToRunMod));

    // Now that we've generated code for this module, take them optimized code
    // and mark the definitions as available externally. We'll link them into
    // future modules this way so that they can be inlined.

    for (auto &F : Consumer->getModule()->functions())
      if (!F.isDeclaration())
        F.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);

    for (auto &GV : Consumer->getModule()->global_values())
      if (!GV.isDeclaration()) {
        if (GV.hasAppendingLinkage())
          cast<GlobalVariable>(GV).setInitializer(nullptr);
        else if (isa<GlobalAlias>(GV))
          // Aliases cannot have externally-available linkage, so give them
          // private linkage.
          GV.setLinkage(llvm::GlobalValue::PrivateLinkage);
        else
          GV.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
      }

    // OverrideFromSrc is needed here too, otherwise globals marked available_externally are not considered.
    if (Linker::linkModules(*RunningMod, Consumer->takeModule(),
                            Linker::Flags::OverrideFromSrc))
      fatal(PassPipeline);

    StringRef StatsFile = Invocation->getFrontendOpts().StatsFile;
    if (!StatsFile.empty()) {
      std::string VariantStatsFile = llvm::Twine(StatsFile)
                                    .concat(llvm::Twine('.'))
                                    .concat(llvm::Twine(FuncMap[Idx]->getName()))
                                    .concat(llvm::Twine('_'))
                                    .concat(llvm::Twine(Idx))
                                    .concat(llvm::Twine('.'))
                                    .concat(llvm::Twine(VariantIdx))
                                    .str();

      std::error_code EC;
      auto StatS = llvm::make_unique<llvm::raw_fd_ostream>(VariantStatsFile, EC,
                                                          llvm::sys::fs::F_Text);
      if (EC) {
        Diagnostics->Report(diag::warn_fe_unable_to_open_stats_file)
            << VariantStatsFile << EC.message();
      } else {
        llvm::PrintStatisticsJSON(*StatS);
      }
    }

    auto AnonStructIds = Consumer->getCodeGenerator()->CGM().getCXXABI().getMangleContext().getAnonStructIds();
    Consumer->Initialize(*Ctx);
    Consumer->getCodeGenerator()->CGM().getCXXABI().getMangleContext().setAnonStructIds(AnonStructIds);

    auto SpecSymbol = CJ->findSymbol(SMName);
    assert(SpecSymbol && "Can't find the specialization just generated?");

    if (!SpecSymbol.getAddress())
      fatal(PassPipeline);

    return (void *) llvm::cantFail(SpecSymbol.getAddress());
  }
};

llvm::sys::SmartMutex<false> Mutex;
bool InitializedTarget = false;
llvm::DenseMap<const void *, std::unique_ptr<CompilerData>> TUCompilerData;

struct InstInfo {
  InstInfo(const char *InstKey, const void *NTTPValues,
           unsigned NTTPValuesSize, const char **TypeStrings,
           unsigned TypeStringsCnt)
    : Key(InstKey),
      NTArgs(StringRef((const char *) NTTPValues, NTTPValuesSize)) {
    for (unsigned i = 0, e = TypeStringsCnt; i != e; ++i)
      TArgs.push_back(StringRef(TypeStrings[i]));
  }

  InstInfo(const StringRef &R) : Key(R) { }

  // The instantiation key (these are always constants, so we don't need to
  // allocate storage for them).
  StringRef Key;

  // The buffer of non-type arguments (this is packed).
  SmallString<16> NTArgs;

  // Vector of string type names.
  SmallVector<SmallString<32>, 1> TArgs;
};

struct ThisInstInfo {
  ThisInstInfo(const char *InstKey, const void *NTTPValues,
               unsigned NTTPValuesSize, const char **TypeStrings,
               unsigned TypeStringsCnt)
    : InstKey(InstKey), NTTPValues(NTTPValues), NTTPValuesSize(NTTPValuesSize),
      TypeStrings(TypeStrings), TypeStringsCnt(TypeStringsCnt) {}

  const char *InstKey;

  const void *NTTPValues;
  unsigned NTTPValuesSize;

  const char **TypeStrings;
  unsigned TypeStringsCnt;
};

struct InstMapInfo {
  static inline InstInfo getEmptyKey() {
    return InstInfo(DenseMapInfo<StringRef>::getEmptyKey());
  }

  static inline InstInfo getTombstoneKey() {
    return InstInfo(DenseMapInfo<StringRef>::getTombstoneKey());
  }

  static unsigned getHashValue(const InstInfo &II) {
    using llvm::hash_code;
    using llvm::hash_combine;
    using llvm::hash_combine_range;

    hash_code h = hash_combine_range(II.Key.begin(), II.Key.end());
    h = hash_combine(h, hash_combine_range(II.NTArgs.begin(),
                                           II.NTArgs.end()));
    for (auto &TA : II.TArgs)
      h = hash_combine(h, hash_combine_range(TA.begin(), TA.end()));

    return (unsigned) h;
  }
  
  static unsigned getHashValue(const ThisInstInfo &TII) {
    using llvm::hash_code;
    using llvm::hash_combine;
    using llvm::hash_combine_range;

    hash_code h =
      hash_combine_range(TII.InstKey, TII.InstKey + std::strlen(TII.InstKey));
    h = hash_combine(h, hash_combine_range((const char *) TII.NTTPValues,
                                           ((const char *) TII.NTTPValues) +
                                             TII.NTTPValuesSize));
    for (unsigned int i = 0, e = TII.TypeStringsCnt; i != e; ++i)
      h = hash_combine(h,
                       hash_combine_range(TII.TypeStrings[i],
                                          TII.TypeStrings[i] +
                                            std::strlen(TII.TypeStrings[i])));

    return (unsigned) h;
  }

  static bool isEqual(const InstInfo &LHS, const InstInfo &RHS) {
    return LHS.Key    == RHS.Key &&
           LHS.NTArgs == RHS.NTArgs &&
           LHS.TArgs  == RHS.TArgs;
  }

  static bool isEqual(const ThisInstInfo &LHS, const InstInfo &RHS) {
    return isEqual(RHS, LHS);
  }

  static bool isEqual(const InstInfo &II, const ThisInstInfo &TII) {
    if (II.Key != StringRef(TII.InstKey))
      return false;
    if (II.NTArgs != StringRef((const char *) TII.NTTPValues,
                               TII.NTTPValuesSize))
      return false;
    if (II.TArgs.size() != TII.TypeStringsCnt)
      return false;
    for (unsigned int i = 0, e = TII.TypeStringsCnt; i != e; ++i)
      if (II.TArgs[i] != StringRef(TII.TypeStrings[i]))
        return false;

    return true; 
  }
};

llvm::sys::SmartMutex<false> IMutex;
llvm::DenseMap<InstInfo, void *, InstMapInfo> Instantiations;

} // anonymous namespace

extern "C"
#ifdef _MSC_VER
__declspec(dllexport)
#endif
void *__clang_jit(const void *CmdArgs, unsigned CmdArgsLen,
                  const void *ASTBuffer, size_t ASTBufferSize,
                  const void *IRBuffer, size_t IRBufferSize,
                  const void **LocalPtrs, unsigned LocalPtrsCnt,
                  const void **LocalDbgPtrs, unsigned LocalDbgPtrsCnt,
                  const DevData *DeviceData, unsigned DevCnt,
                  const void *NTTPValues, unsigned NTTPValuesSize,
                  const char **TypeStrings, unsigned TypeStringsCnt,
                  const char *InstKey, unsigned Idx) {
  {
    llvm::MutexGuard Guard(IMutex);
    auto II =
      Instantiations.find_as(ThisInstInfo(InstKey, NTTPValues, NTTPValuesSize,
                                          TypeStrings, TypeStringsCnt));
    if (II != Instantiations.end())
      return II->second;

    if (OptimizationPasses.size() == 0)
      InitOptimizationPasses();
  }

  llvm::MutexGuard Guard(Mutex);

  llvm::errs() << "JIT: compiling new variant ...\n";

  if (!InitializedTarget) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    LCtx.reset(new LLVMContext);

    InitializedTarget = true;
  }

  CompilerData *CD;
  auto TUCDI = TUCompilerData.find(ASTBuffer);
  if (TUCDI == TUCompilerData.end()) {
    CD = new CompilerData(CmdArgs, CmdArgsLen, ASTBuffer, ASTBufferSize,
                          IRBuffer, IRBufferSize, LocalPtrs, LocalPtrsCnt,
                          LocalDbgPtrs, LocalDbgPtrsCnt, DeviceData, DevCnt);
    TUCompilerData[ASTBuffer].reset(CD);
  } else {
    CD = TUCDI->second.get();
  }

  void *FPtr = CD->resolveFunction(NTTPValues, TypeStrings, Idx);

  {
    CurrentKernel++;
    llvm::MutexGuard Guard(IMutex);
    Instantiations[InstInfo(InstKey, NTTPValues, NTTPValuesSize,
                            TypeStrings, TypeStringsCnt)] = FPtr;
  }

  return FPtr;
}

