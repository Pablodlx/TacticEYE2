"""
Script de Verificación de Integración de Heatmaps
==================================================

Verifica que el sistema de heatmaps esté correctamente integrado en app.py
"""

import sys
from pathlib import Path

def check_imports():
    """Verifica que los imports estén correctos"""
    print("1. Verificando imports...")
    try:
        from modules.field_heatmap_system import (
            FIELD_POINTS,
            HeatmapAccumulator,
            estimate_homography_with_flip_resolution
        )
        print("   ✓ Imports del sistema de heatmaps OK")
        return True
    except Exception as e:
        print(f"   ✗ Error importando sistema de heatmaps: {e}")
        return False

def check_batch_processor():
    """Verifica que BatchProcessor tenga el acumulador"""
    print("\n2. Verificando BatchProcessor...")
    try:
        from modules.batch_processor import BatchProcessor
        
        # Verificar que tiene los imports necesarios
        import inspect
        source = inspect.getsource(BatchProcessor)
        
        checks = {
            'import HeatmapAccumulator': 'HeatmapAccumulator' in source,
            'import estimate_homography_with_flip_resolution': 'estimate_homography_with_flip_resolution' in source,
            'self.heatmap_accumulator': 'self.heatmap_accumulator' in source,
            'add_frame': 'add_frame' in source
        }
        
        all_ok = True
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"   {status} {check}: {result}")
            if not result:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"   ✗ Error verificando BatchProcessor: {e}")
        return False

def check_app_integration():
    """Verifica que app.py tenga los imports y endpoint modificado"""
    print("\n3. Verificando app.py...")
    try:
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
            print("   ✗ app.py no encontrado")
            return False
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        checks = {
            'import HeatmapAccumulator': 'HeatmapAccumulator' in content,
            'import FIELD_POINTS': 'FIELD_POINTS' in content,
            'heatmap_flip_key': 'heatmap_flip_key' in content,
            'team_{team_id}_heatmap_flip': 'team_{team_id}_heatmap_flip' in content
        }
        
        all_ok = True
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"   {status} {check}: {result}")
            if not result:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"   ✗ Error verificando app.py: {e}")
        return False

def check_export_function():
    """Verifica que export_spatial_heatmaps guarde los nuevos heatmaps"""
    print("\n4. Verificando función de exportación...")
    try:
        from modules.batch_processor import export_spatial_heatmaps
        import inspect
        source = inspect.getsource(export_spatial_heatmaps)
        
        checks = {
            'heatmap_flip_0': 'heatmap_flip_0' in source,
            'heatmap_flip_1': 'heatmap_flip_1' in source,
            'get_heatmap': 'get_heatmap' in source,
            'team_0_heatmap_flip': 'team_0_heatmap_flip' in source,
            'team_1_heatmap_flip': 'team_1_heatmap_flip' in source
        }
        
        all_ok = True
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"   {status} {check}: {result}")
            if not result:
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"   ✗ Error verificando export_spatial_heatmaps: {e}")
        return False

def main():
    print("=" * 70)
    print("VERIFICACIÓN DE INTEGRACIÓN DEL SISTEMA DE HEATMAPS")
    print("=" * 70)
    
    results = []
    
    results.append(("Imports", check_imports()))
    results.append(("BatchProcessor", check_batch_processor()))
    results.append(("app.py", check_app_integration()))
    results.append(("Exportación", check_export_function()))
    
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ¡INTEGRACIÓN COMPLETA Y VERIFICADA!")
        print("\nEl sistema de heatmaps con resolución de flip está:")
        print("  ✓ Importado correctamente")
        print("  ✓ Integrado en BatchProcessor")
        print("  ✓ Integrado en app.py")
        print("  ✓ Configurado para exportar heatmaps mejorados")
        print("\nPróximos pasos:")
        print("  1. Ejecutar app.py")
        print("  2. Subir un video")
        print("  3. Verificar que se generen heatmaps con '_flip' en el NPZ")
        print("  4. Verificar que el endpoint /api/heatmap use los nuevos heatmaps")
        return 0
    else:
        print("\n⚠️  INTEGRACIÓN INCOMPLETA")
        print("\nHay componentes que no pasaron la verificación.")
        print("Revisa los mensajes de error arriba.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
